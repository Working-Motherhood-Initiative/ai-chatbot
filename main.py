from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import openai
import os
from dotenv import load_dotenv
import pdfminer.high_level
from docx import Document
from job_fetcher import find_jobs_from_sentence, preload_job_embeddings, get_all_jobs
from datetime import datetime
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
from pydantic import BaseModel
from labour_law_rag import get_rag_instance, initialize_rag_system
from assessment_questions import (
    ASSESSMENT_QUESTIONS,
    calculate_assessment_scores,
    generate_assessment_feedback,
    get_assessment_instructions,
    validate_responses
)
import threading

class LabourLawQuery(BaseModel):
    query: str
    country: Optional[str] = None
    user_context: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None

class AssessmentResponse(BaseModel):
    career_readiness: Dict[str, str]
    work_life_balance: Dict[str, str]

security = HTTPBearer()
API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    raise Exception("API_TOKEN not found in environment variables")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        logger.warning("Invalid token attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials

log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Motherboard Career Assistant API", 
    version="1.0.0",
    description="API for the Motherboard Career Assistant - helping mothers navigate work and career"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

try:
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise

_initialization_complete = False
_initialization_error = None

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# PRIVACY PROTECTION FUNCTIONS
def remove_personal_info_from_cv(cv_text: str) -> str:
    try:
        if not cv_text or not cv_text.strip():
            logger.warning("Empty CV text provided for privacy protection")
            return ""
            
        cleaned_text = cv_text
        
        # Remove email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        cleaned_text = re.sub(email_pattern, '[EMAIL REMOVED]', cleaned_text)
        
        # Remove phone numbers
        phone_patterns = [
            r'\+?\d{1,4}[\s\-\(\)]?\(?\d{1,4}\)?[\s\-]?\d{1,4}[\s\-]?\d{1,9}',
            r'\b\d{3}[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b',
            r'\(\d{3}\)\s?\d{3}[\s\-]?\d{4}',
        ]
        for pattern in phone_patterns:
            cleaned_text = re.sub(pattern, '[PHONE REMOVED]', cleaned_text)
        
        # Remove addresses
        address_patterns = [
            r'\b\d+\s+[A-Za-z\s]+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Place|Pl)\b.*',
            r'\b\d+[A-Za-z]?\s+[A-Za-z\s]+\d{5}(-\d{4})?\b',
        ]
        for pattern in address_patterns:
            cleaned_text = re.sub(pattern, '[ADDRESS REMOVED]', cleaned_text, flags=re.IGNORECASE)
        
        # Remove social security numbers
        ssn_pattern = r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
        cleaned_text = re.sub(ssn_pattern, '[ID REMOVED]', cleaned_text)
        
        # Remove LinkedIn URLs
        linkedin_pattern = r'https?://(?:www\.)?linkedin\.com/in/[^\s]+'
        cleaned_text = re.sub(linkedin_pattern, '[LINKEDIN PROFILE]', cleaned_text)
        
        # Clean up extra whitespace
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
        
        # Add privacy notice
        privacy_notice = "=== PRIVACY-PROTECTED CV ANALYSIS ===\n[Personal identifying information has been removed for privacy protection]\n\n"
        cleaned_text = privacy_notice + cleaned_text.strip()
        
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Privacy protection failed: {e}")
        raise HTTPException(status_code=500, detail="Privacy protection system failed")

def validate_file_size(file: UploadFile):
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 5MB.")

# Utility functions
def extract_text_from_pdf(file):
    try:
        return pdfminer.high_level.extract_text(file)
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF")

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {e}")
        raise HTTPException(status_code=400, detail="Failed to extract text from DOCX")

def get_ai_response(messages: List[Dict], max_tokens: int = 500) -> str:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return "I'm sorry, I'm having trouble processing your request right now. Please try again later."

def calculate_semantic_similarity(cv_text: str, job_description: str) -> float:
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=500,
            ngram_range=(1, 2)
        )
        documents = [cv_text[:2000], job_description[:1000]]
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity * 100
    except Exception as e:
        logger.warning(f"Semantic similarity calculation failed: {e}")
        return 50.0

# Background initialization
def initialize_background():
    global _initialization_complete, _initialization_error
    try:
        logger.info("Background initialization started...")
        
        # Load job data
        logger.info("Starting job data initialization...")
        preload_job_embeddings()
        jobs = get_all_jobs()
        logger.info(f"Successfully loaded {len(jobs)} jobs")
        
        # Initialize Labour Law RAG System
        logger.info("Initializing Labour Law RAG system...")
        try:
            doc_count = initialize_rag_system(
                pdf_directory="labour_laws",
                force_reload=False,
                use_gdrive=True
            )
            logger.info(f"Successfully loaded {doc_count} labour law document chunks")
        except Exception as e:
            logger.error(f"Labour Law RAG initialization failed: {e}")
            _initialization_error = str(e)
        
        _initialization_complete = True
        logger.info("Background initialization completed!")
        
    except Exception as e:
        logger.error(f"Background initialization error: {e}")
        _initialization_error = str(e)
        _initialization_complete = True

@app.on_event("startup")
async def startup_event():
    try:
        # Test privacy protection
        test_text = "John Doe john@email.com (555) 123-4567 Professional Summary: Software developer"
        cleaned = remove_personal_info_from_cv(test_text)
        if "john@email.com" in cleaned:
            raise Exception("Privacy protection system failed startup test")
        logger.info("Privacy protection startup test passed")
        
        logger.info("API server starting...")
        
        # Start initialization in background thread
        init_thread = threading.Thread(target=initialize_background, daemon=True)
        init_thread.start()
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise



@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        rag = get_rag_instance()
        stats = rag.get_system_stats()
        gdrive_status = "Enabled" if stats.get('gdrive_enabled') else "Disabled"
    except:
        gdrive_status = "Unknown"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Motherboard Career Assistant API</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
            h1 {{ color: #667eea; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Motherboard Career Assistant API</h1>
            <p>Status: Running with Privacy Protection</p>
            <p>Google Drive Integration: {gdrive_status}</p>
            <h2>Available Endpoints</h2>
            <ul>
                <li>GET /jobs - Get all available jobs</li>
                <li>POST /labour-law-query - Ask labour law questions</li>
                <li>GET /assessment-questions - Get assessment questions</li>
                <li>POST /momfit-assessment - Submit assessment</li>
                <li>GET /health - Health check</li>
                <li>GET /docs - API documentation</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/init-status")
async def initialization_status():
    return JSONResponse({
        "initialization_complete": _initialization_complete,
        "error": _initialization_error,
        "timestamp": datetime.now().isoformat()
    })

@app.get("/jobs")
async def get_all_available_jobs(token: str = Depends(verify_token)):
    try:
        jobs = get_all_jobs()
        logger.info(f"Fetching all {len(jobs)} available jobs")
        
        complete_jobs = []
        for idx, job in enumerate(jobs):
            job_dict = dict(job)
            
            if 'embedding' in job_dict:
                del job_dict['embedding']
            
            description = (
                job_dict.get("Job Description (Brief Summary)") or 
                job_dict.get("Job Description") or 
                "N/A"
            )
            
            complete_job = {
                "job_id": idx,
                "job_title": job_dict.get("Job Title", "N/A"),
                "company_name": job_dict.get("Company Name", job_dict.get("Company", "N/A")),
                "job_type": job_dict.get("Job Type", "N/A"),
                "industry": job_dict.get("Industry", "N/A"),
                "job_description": description,
                "location": job_dict.get("Location", "N/A"),
                "application_link": job_dict.get("Application Link or Email", "N/A"),
                "application_deadline": job_dict.get("Application Deadline", "N/A"),
                "skills_required": job_dict.get("Skills Required", "N/A"),
            }
            
            complete_jobs.append(complete_job)
            
        return JSONResponse({
            "jobs": complete_jobs,
            "total_jobs": len(complete_jobs),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching jobs: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch jobs", "details": str(e)}
        )

@app.post("/labour-law-query")
async def labour_law_query(request_data: LabourLawQuery, token: str = Depends(verify_token)):
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
            }
        })

    except Exception as e:
        logger.error(f"Labour law query error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to process labour law query"}
        )

@app.get("/assessment-questions")
async def get_assessment_questions(token: str = Depends(verify_token)):
    try:
        instructions = get_assessment_instructions()
        
        return JSONResponse({
            "questions": ASSESSMENT_QUESTIONS,
            "instructions": instructions,
            "metadata": {
                "total_questions": len(ASSESSMENT_QUESTIONS["career_readiness"]) + len(ASSESSMENT_QUESTIONS["work_life_balance"]),
                "estimated_time_minutes": 7
            }
        })
        
    except Exception as e:
        logger.error(f"Assessment questions error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve assessment questions"}
        )

@app.post("/momfit-assessment")
async def submit_momfit_assessment(responses: AssessmentResponse,token: str = Depends(verify_token)):
    try:
        is_valid, error_message = validate_responses(responses.dict())
        if not is_valid:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid responses", "details": error_message}
            )
        
        scores = calculate_assessment_scores(responses.dict())
        feedback = generate_assessment_feedback(scores)
        
        return JSONResponse({
            "scores": {
                "career_readiness": scores["career_readiness_score"],
                "work_life_balance": scores["work_life_balance_score"], 
                "overall": scores["overall_score"]
            },
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"MomFit assessment error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to process assessment"}
        )

@app.get("/health")
async def health_check():
    try:
        test_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        openai_status = "Connected"
    except:
        openai_status = "Error"
    
    try:
        jobs = get_all_jobs()
        job_data_status = f"{len(jobs)} jobs loaded"
    except:
        job_data_status = "Job data error"
    
    try:
        test_cv = "John Doe john@email.com Professional Summary: Developer"
        cleaned = remove_personal_info_from_cv(test_cv)
        privacy_status = "Active" if "[EMAIL REMOVED]" in cleaned else "Failed"
    except:
        privacy_status = "Error"
    
    try:
        rag = get_rag_instance()
        stats = rag.get_system_stats()
        labour_law_status = f"Active - {stats['total_chunks']} chunks" if stats['documents_loaded'] else "Not initialized"
        gdrive_info = {
            "enabled": stats.get('gdrive_enabled', False),
            "local_cache": stats.get('local_cache_exists', False)
        }
    except:
        labour_law_status = "Error"
        gdrive_info = {"enabled": False, "local_cache": False}
    
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "Running",
            "openai": openai_status,
            "job_data": job_data_status,
            "privacy_protection": privacy_status,
            "labour_law_rag": labour_law_status
        },
        "google_drive": gdrive_info
    })

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
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
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Motherboard Career Assistant API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)