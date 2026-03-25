from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends, status, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import asyncio
import logging
from datetime import datetime
from typing import Optional
from feedback_generator import CareerFeedbackGenerator
from schemas import LabourLawQuery, AssessmentResponse
from session_manager import SessionStore
from privacy_protection import remove_personal_info_from_cv, log_privacy_protection
from cv_analyzer import calculate_cv_job_match_hybrid, get_match_ranking
from file_utils import extract_text_from_pdf, extract_text_from_docx, validate_file_size
from llm_utils import get_ai_response
from config import (
    setup_logging, load_environment, initialize_openai_client,
    cleanup_sessions_periodically, test_privacy_protection,
    start_background_initialization, _initialization_complete, _initialization_error
)
from job_fetcher import find_jobs_from_sentence, get_all_jobs
from labour_law_rag import get_rag_instance
from assessment_questions import (
    ASSESSMENT_QUESTIONS,
    calculate_assessment_scores,
    generate_assessment_feedback,
    get_assessment_instructions,
    validate_responses
)

load_dotenv = __import__('dotenv').load_dotenv
load_dotenv()

logger = setup_logging()

session_store = SessionStore()

security = HTTPBearer()
API_TOKEN = load_environment()

openai_client = initialize_openai_client()
feedback_generator = CareerFeedbackGenerator(openai_client)


app = FastAPI(
    title="Motherboard Career Assistant API",
    version="2.0.0",
    description="Career Guidance API for Motherboard"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dragonfly-chihuahua-alhg.squarespace.com",
        "https://ai-chatbot-4bqx.onrender.com",
        "https://workingmotherhoodinitiative.org",
        "localhost",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        logger.warning("Invalid token attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials


async def verify_session(
    authorization: Optional[str] = Header(None)
) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing session token. Please refresh the page."
        )

    session_id = authorization.replace("Bearer ", "")

    if not session_store.validate_session(session_id):
        raise HTTPException(
            status_code=401,
            detail="Session expired. Please refresh the page."
        )

    return session_id


@app.on_event("startup")
async def startup_event():
    try:
        test_privacy_protection(remove_personal_info_from_cv)
        logger.info("API server starting...")
        logger.info("Session management initialized")

        asyncio.create_task(cleanup_sessions_periodically(session_store))

        logger.info("Running heavy initialization in background thread...")
        start_background_initialization()

        logger.info("Job system ready - will load from Google Sheet on demand")

    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise


@app.get("/")
async def root():
    return {"message": "Motherboard Career Assistant API", "version": "2.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/init-status")
async def initialization_status():
    return JSONResponse({
        "initialization_complete": _initialization_complete,
        "error": _initialization_error,
        "timestamp": datetime.now().isoformat()
    })


@app.post("/create-session")
async def create_session():
    session_id = session_store.create_session()
    return {
        "session_id": session_id,
        "expires_in_hours": 2,
        "message": "Session created successfully"
    }


@app.get("/api-info")
async def api_info():
    return JSONResponse({
        "api_name": "Motherboard Career Assistant API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": [
            "POST /create-session",
            "POST /welcome",
            "GET /jobs",
            "POST /search-jobs",
            "POST /cv-job-match",
            "GET /assessment-questions",
            "POST /momfit-assessment",
            "GET /assessment-stats",
            "POST /labour-law-query",
            "GET /labour-law-countries",
            "POST /feedback",
            "POST /admin/reload-vectorstore"
        ]
    })


@app.post("/welcome")
async def welcome_user(request: Request, session: str = Depends(verify_session)):
    try:
        data = await request.json()
    except Exception:
        data = {}

    user_name = data.get("name", "there")
    welcome_message = f"Welcome back, {user_name}! How can I help you today?"

    return JSONResponse({"response": welcome_message})


@app.get("/jobs")
async def get_all_available_jobs(session: str = Depends(verify_session)):
    try:
        jobs = get_all_jobs()
        logger.info(f"Fetching all {len(jobs)} available jobs")

        unique_locations = set()
        unique_industries = set()
        unique_job_types = set()

        complete_jobs = []
        for idx, job in enumerate(jobs):
            job_dict = dict(job)

            if 'embedding' in job_dict:
                del job_dict['embedding']

            description = (
                job_dict.get("Job Description (Brief Summary)") or
                job_dict.get("Job Description (Brief Summary)  ") or
                job_dict.get("Job Description") or
                "N/A"
            )

            location = job_dict.get("Location", "N/A")
            if location and location != "N/A":
                for part in location.split(','):
                    clean_location = part.strip()
                    if clean_location and clean_location != 'N/A':
                        unique_locations.add(clean_location)

            industry = job_dict.get("Industry", "N/A")
            if industry and industry != "N/A":
                unique_industries.add(industry.strip())

            job_type = job_dict.get("Job Type", "N/A")
            if job_type and job_type != "N/A":
                unique_job_types.add(job_type.strip())

            complete_jobs.append({
                "job_id": idx,
                "job_title": job_dict.get("Job Title", "N/A"),
                "company_name": job_dict.get("Company Name", job_dict.get("Company", "N/A")),
                "job_type": job_type,
                "industry": industry,
                "job_description": description,
                "location": location,
                "application_link": job_dict.get("Application Link or Email", job_dict.get("Application Link", "N/A")),
                "application_deadline": job_dict.get("Application Deadline", "N/A"),
                "skills_required": job_dict.get("Skills Required", "N/A"),
                "additional_fields": {k: v for k, v in job_dict.items() if k not in [
                    "Job Title", "Company Name", "Company", "Job Type", "Industry",
                    "Job Description (Brief Summary)", "Job Description (Brief Summary)  ",
                    "Job Description", "Location", "Application Link or Email",
                    "Application Link", "Application Deadline", "Skills Required"
                ]}
            })

        sorted_locations = sorted(list(unique_locations))
        sorted_industries = sorted(list(unique_industries))
        sorted_job_types = sorted(list(unique_job_types))

        logger.info(f"Successfully formatted {len(complete_jobs)} jobs")

        return JSONResponse({
            "jobs": complete_jobs,
            "total_jobs": len(complete_jobs),
            "message": f"Showing all {len(complete_jobs)} available job(s).",
            "available_filters": {
                "countries": sorted_locations,
                "industries": sorted_industries,
                "job_types": sorted_job_types
            },
            "suggestions": [
                "Analyze your CV against a job",
                "Get CV feedback",
                "Take MomFit Assessment"
            ] if complete_jobs else [
                "No jobs available at the moment",
                "Check back soon for new opportunities"
            ]
        })

    except Exception as e:
        logger.error(f"Error fetching jobs: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch jobs", "details": str(e)}
        )


@app.post("/search-jobs")
async def search_jobs(request: Request, session: str = Depends(verify_session)):
    try:
        data = await request.json()
        user_query = data.get("query", "")
        country_filter = data.get("country", "").strip()
        industry_filter = data.get("industry", "").strip()
        job_type_filter = data.get("job_type", "").strip()

        logger.info(f"Search - Query: '{user_query}', Country: '{country_filter}', Industry: '{industry_filter}', Type: '{job_type_filter}'")

        jobs = find_jobs_from_sentence(user_query) if user_query.strip() else get_all_jobs()

        unique_locations = set()
        unique_industries = set()
        unique_job_types = set()

        complete_jobs = []
        for idx, job in enumerate(jobs):
            job_dict = dict(job)

            if 'embedding' in job_dict:
                del job_dict['embedding']

            description = (
                job_dict.get("Job Description (Brief Summary)") or
                job_dict.get("Job Description (Brief Summary)  ") or
                job_dict.get("Job Description") or
                "N/A"
            )

            location = job_dict.get("Location", "N/A")
            industry = job_dict.get("Industry", "N/A")
            job_type = job_dict.get("Job Type", "N/A")

            if location and location != "N/A":
                for part in location.split(','):
                    clean_location = part.strip()
                    if clean_location and clean_location != 'N/A':
                        unique_locations.add(clean_location)

            if industry and industry != "N/A":
                unique_industries.add(industry.strip())

            if job_type and job_type != "N/A":
                unique_job_types.add(job_type.strip())

            complete_jobs.append({
                "job_id": idx,
                "job_title": job_dict.get("Job Title", "N/A"),
                "company_name": job_dict.get("Company Name", job_dict.get("Company", "N/A")),
                "job_type": job_type,
                "industry": industry,
                "job_description": description,
                "location": location,
                "application_link": job_dict.get("Application Link or Email", job_dict.get("Application Link", "N/A")),
                "application_deadline": job_dict.get("Application Deadline", "N/A"),
                "skills_required": job_dict.get("Skills Required", "N/A"),
            })

        if country_filter:
            complete_jobs = [
                job for job in complete_jobs
                if any(country_filter == loc.strip() for loc in job.get("location", "").split(','))
            ]

        if industry_filter:
            complete_jobs = [
                job for job in complete_jobs
                if industry_filter.lower() in str(job.get("industry", "")).lower()
            ]

        if job_type_filter:
            complete_jobs = [
                job for job in complete_jobs
                if job_type_filter.lower() in str(job.get("job_type", "")).lower()
            ]

        filters_applied = []
        if user_query:
            filters_applied.append(f"query '{user_query}'")
        if country_filter:
            filters_applied.append(f"country '{country_filter}'")
        if industry_filter:
            filters_applied.append(f"industry '{industry_filter}'")
        if job_type_filter:
            filters_applied.append(f"type '{job_type_filter}'")

        filter_text = " and ".join(filters_applied) if filters_applied else "no filters"

        return JSONResponse({
            "jobs": complete_jobs,
            "total_found": len(complete_jobs),
            "filters_applied": {
                "query": user_query or None,
                "country": country_filter or None,
                "industry": industry_filter or None,
                "job_type": job_type_filter or None
            },
            "message": f"Found {len(complete_jobs)} job(s) matching {filter_text}.",
            "available_filters": {
                "countries": sorted(list(unique_locations)),
                "industries": sorted(list(unique_industries)),
                "job_types": sorted(list(unique_job_types))
            }
        })

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Search failed", "details": str(e)}
        )


@app.post("/cv-job-match")
async def analyze_cv_job_match_hybrid_endpoint(
    file: UploadFile = File(...),
    jobTitle: str = Form(...),
    company: str = Form(...),
    jobDescription: str = Form(...),
    skillsRequired: str = Form(default=""),
    session: str = Depends(verify_session)
):
    try:
        validate_file_size(file)

        if file.filename.endswith(".pdf"):
            original_cv_content = extract_text_from_pdf(file.file)
        elif file.filename.endswith(".docx"):
            original_cv_content = extract_text_from_docx(file.file)
        else:
            raise HTTPException(status_code=400, detail="File must be PDF or DOCX")

        if not original_cv_content or len(original_cv_content.strip()) == 0:
            raise HTTPException(status_code=400, detail="Could not extract text from file")

        cleaned_cv = remove_personal_info_from_cv(original_cv_content)
        privacy_validation = log_privacy_protection(original_cv_content, cleaned_cv, "cv-job-match")

        logger.info(f"Analyzing CV for job: {jobTitle} at {company}")

        match_result = calculate_cv_job_match_hybrid(
            cleaned_cv,
            jobDescription,
            jobTitle,
            skillsRequired
        )
        ranking = get_match_ranking(match_result["overall_match"])

        combined_job_info = f"{jobDescription}\n\nRequired Skills: {skillsRequired}".strip()

        feedback_gen = CareerFeedbackGenerator(openai_client)
        simple_feedback = feedback_gen.generate_detailed_feedback(
            match_result=match_result,
            job_title=jobTitle,
            company=company,
            cv_content=cleaned_cv,
            job_description=combined_job_info
        )

        return JSONResponse({
            "status": "success",
            "overall_match_percentage": simple_feedback["match_percentage"],
            "match_level": simple_feedback["match_level"],
            "match_color": ranking["color"],
            "strengths": simple_feedback["strengths"],
            "improvements": simple_feedback["improvements"],
            "scores": simple_feedback["scores"],
            "privacy_protection": {
                "status": privacy_validation["privacy_check_passed"],
                "reduction_percentage": privacy_validation["statistics"]["reduction_percentage"]
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CV analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing CV: {str(e)}")


@app.get("/assessment-questions")
async def get_assessment_questions(session: str = Depends(verify_session)):
    try:
        return JSONResponse({
            "questions": ASSESSMENT_QUESTIONS,
            "assessment_sections": {
                "career_readiness": {
                    "title": "Career Readiness Assessment",
                    "focus_areas": ["Career goals", "Skills", "Experience", "Professional development"],
                    "question_count": len(ASSESSMENT_QUESTIONS.get("career_readiness", []))
                },
                "work_life_balance": {
                    "title": "Work-Life Balance Assessment",
                    "focus_areas": ["Time management", "Support systems", "Stress management"],
                    "question_count": len(ASSESSMENT_QUESTIONS.get("work_life_balance", []))
                }
            },
            "total_questions": sum(len(q) for q in ASSESSMENT_QUESTIONS.values()),
            "estimated_time_minutes": 15
        })
    except Exception as e:
        logger.error(f"Error getting assessment questions: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to retrieve questions"})


@app.post("/momfit-assessment")
async def momfit_assessment(request: Request, session: str = Depends(verify_session)):
    try:
        data = await request.json()
        responses = data.get("responses", {})

        if not responses:
            raise HTTPException(status_code=400, detail="responses are required")

        validation_result = validate_responses(responses)
        if not validation_result["valid"]:
            return JSONResponse(
                status_code=400,
                content={"error": validation_result["error"]}
            )

        scores = calculate_assessment_scores(responses)
        feedback = generate_assessment_feedback(scores)

        return JSONResponse({
            "status": "success",
            "scores": scores,
            "feedback": feedback,
            "recommendations": [
                "Based on your career readiness score, focus on developing key skills",
                "Prioritize work-life balance for sustainable career growth",
                "Schedule regular career development activities"
            ]
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Assessment error: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")


@app.get("/assessment-stats")
async def assessment_statistics(session: str = Depends(verify_session)):
    try:
        return JSONResponse({
            "assessment_stats": {
                "total_sections": 2,
                "career_readiness": {
                    "count": len(ASSESSMENT_QUESTIONS["career_readiness"]),
                    "focus_areas": ["Career goals", "Skills", "Experience"]
                },
                "work_life_balance": {
                    "count": len(ASSESSMENT_QUESTIONS["work_life_balance"]),
                    "focus_areas": ["Time management", "Support systems", "Stress management"]
                }
            }
        })
    except Exception as e:
        logger.error(f"Assessment stats error: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to retrieve assessment statistics"})


@app.post("/labour-law-query")
async def labour_law_query(request_data: LabourLawQuery, session: str = Depends(verify_session)):
    try:
        rag = get_rag_instance()

        if not rag.documents_loaded:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Labour law system is not initialized",
                    "message": "Please try again in a moment"
                }
            )

        if not request_data.query or len(request_data.query.strip()) < 5:
            return JSONResponse(
                status_code=400,
                content={"error": "Please provide a valid question (minimum 5 characters)"}
            )

        if request_data.country:
            supported_countries = rag.get_supported_countries()
            if request_data.country not in supported_countries:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Country '{request_data.country}' not supported",
                        "supported_countries": supported_countries
                    }
                )

        logger.info(f"Labour law query: '{request_data.query}' for country: {request_data.country or 'All'}")

        result = rag.generate_answer(
            query=request_data.query,
            country=request_data.country,
            user_context=request_data.user_context or "working mother seeking labour rights information",
            chat_history=request_data.chat_history or []
        )

        return JSONResponse({
            "answer": result["answer"],
            "sources": result["sources"],
            "chat_history": result.get("chat_history", []),
            "metadata": {
                "country": result.get("country"),
                "confidence": result.get("confidence"),
                "query": request_data.query,
                "timestamp": datetime.now().isoformat()
            },
            "next_steps": [
                "Ask another labour law question",
                "Search for jobs in your country",
                "Get CV feedback",
                "Take MomFit Assessment"
            ],
            "disclaimer": "This information is for educational purposes. For specific legal advice, consult a qualified employment lawyer."
        })

    except Exception as e:
        logger.error(f"Labour law query error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to process labour law query",
                "message": "Please try again or rephrase your question"
            }
        )


@app.get("/labour-law-countries")
async def get_labour_law_countries(session: str = Depends(verify_session)):
    try:
        rag = get_rag_instance()
        countries = rag.get_supported_countries()
        stats = rag.get_system_stats()

        return JSONResponse({
            "supported_countries": countries,
            "total_countries": len(countries),
            "system_stats": stats,
            "example_queries": [
                "What are my maternity leave rights?",
                "Can I request flexible work hours?",
                "What protection do I have against pregnancy discrimination?",
                "What are my rights if I'm breastfeeding at work?",
                "Can I be fired while on maternity leave?"
            ]
        })

    except Exception as e:
        logger.error(f"Error getting labour law countries: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to retrieve country information"})


@app.post("/feedback")
async def submit_feedback(request: Request, session: str = Depends(verify_session)):
    try:
        data = await request.json()

        rating = data.get("rating")
        comment = data.get("comment", "")
        feature = data.get("feature", "general")
        user_id = data.get("user_id", "anonymous")

        logger.info(f"Feedback received - Rating: {rating}, Feature: {feature}, User: {user_id}")
        logger.info(f"Comment: {comment}")

        return JSONResponse({
            "message": "Thank you for your feedback!",
            "feedback_id": f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to submit feedback"})


@app.post("/admin/reload-vectorstore")
async def admin_reload_vectorstore(token: str = Depends(verify_token)):
    try:
        logger.info("Admin: Manual vectorstore reload initiated...")

        rag = get_rag_instance()
        doc_count = rag.reload_from_gdrive()

        return JSONResponse({
            "status": "success",
            "message": "Vector store reloaded successfully",
            "document_chunks": doc_count,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Admin reload failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to reload vector store", "details": str(e)}
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected error occurred",
            "message": "Please try again later",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Motherboard Career Assistant API (v2)...")
    logger.info("Privacy Protection: ENABLED")
    logger.info("Session Management: ENABLED")
<<<<<<< HEAD
    uvicorn.run(app, host="0.0.0.0", port=8000)
=======
    uvicorn.run(app, host="0.0.0.0", port=8000) 
>>>>>>> a4104fba6114edda40d7e6e7d5b43d28965c7e03
