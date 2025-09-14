from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import secrets
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import openai
import os
from dotenv import load_dotenv
import pdfminer.high_level
from docx import Document
from job_fetcher import find_jobs_from_sentence, preload_job_embeddings, get_all_jobs
from datetime import datetime
import logging
import json
from typing import Optional, Dict, List, Tuple
import re
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import Counter

# Add this import at the top of your main.py file, along with your other imports
from assessment_questions import (
    ASSESSMENT_QUESTIONS,
    calculate_assessment_scores,
    generate_assessment_feedback,
    get_assessment_instructions,
    validate_responses
)

# Add this Pydantic model with your other models (if you don't already have it)
from pydantic import BaseModel
from typing import Dict, Any

class AssessmentResponse(BaseModel):
    career_readiness: Dict[str, str]
    work_life_balance: Dict[str, str]

# Security and authentication
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

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Motherboard Career Assistant API", 
    version="1.0.0",
    description="API for the Motherboard Career Assistant - helping mothers navigate work and career"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "https://dragonfly-chihuahua-alhg.squarespace.com/",
        "https://ai-chatbot-4bqx.onrender.com/",
        "http://localhost:8501",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
try:
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise

# Utility functions
def extract_text_from_pdf(file):
    """Extract text from PDF file"""
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

def extract_skills_from_cv(cv_text: str) -> List[str]:
    """Extract skills from CV text using OpenAI"""
    try:
        messages = [
            {"role": "system", "content": "You are a skill extraction expert. Extract key skills, technologies, and competencies from the CV text. Return only a comma-separated list of skills, no explanations."},
            {"role": "user", "content": f"Extract skills from this CV:\n{cv_text[:2000]}"}
        ]
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=200
        )
        
        skills_text = response.choices[0].message.content.strip()
        skills = [skill.strip() for skill in skills_text.split(',') if skill.strip()]
        return skills[:10]
        
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        return []

def get_ai_response(messages: List[Dict], max_tokens: int = 500) -> str:
    """Get response from OpenAI with error handling"""
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

def calculate_cv_job_match_hybrid(cv_text: str, job_description: str, job_title: str) -> Dict:
    keyword_score = calculate_keyword_match(cv_text, job_description)
    skills_score = calculate_skills_match(cv_text, job_description)
    experience_score = calculate_experience_match(cv_text, job_description, job_title)
    semantic_score = calculate_semantic_similarity(cv_text, job_description)
    
    total_score = (
        keyword_score * 0.30 +
        skills_score * 0.25 +
        experience_score * 0.25 +
        semantic_score * 0.20
    )

    missing_keywords = find_missing_keywords(cv_text, job_description)
    strengths = identify_strengths(cv_text, job_description)
    
    return {
        "overall_match": round(total_score),
        "breakdown": {
            "keyword_match": round(keyword_score),
            "skills_match": round(skills_score), 
            "experience_match": round(experience_score),
            "semantic_similarity": round(semantic_score)
        },
        "missing_keywords": missing_keywords,
        "strengths": strengths
    }

def calculate_keyword_match(cv_text: str, job_description: str) -> float:
    """Calculate percentage of important job keywords found in CV"""
    job_keywords = extract_important_keywords(job_description)
    
    if not job_keywords:
        return 50.0
    
    cv_lower = cv_text.lower()
    matches = sum(1 for keyword in job_keywords if keyword.lower() in cv_lower)
    
    return (matches / len(job_keywords)) * 100

def extract_important_keywords(text: str) -> List[str]:
    """Extract important keywords from job description"""
    # Technical skills pattern
    tech_pattern = r'\b(python|javascript|java|react|angular|sql|html|css|git|aws|azure|docker|kubernetes|salesforce|hubspot|excel|powerpoint|google analytics|photoshop|illustrator|figma|canva)\b'
    
    # Soft skills pattern  
    soft_pattern = r'\b(leadership|management|communication|teamwork|problem[\-\s]solving|analytical|creative|organized|customer service|project management)\b'
    
    # Industry terms
    industry_pattern = r'\b(marketing|sales|finance|healthcare|education|technology|consulting|administration|operations|human resources)\b'
    
    keywords = []
    for pattern in [tech_pattern, soft_pattern, industry_pattern]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        keywords.extend(matches)
    
    # Extract capitalized terms (likely tools/skills)
    capitalized = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text)
    keywords.extend([cap for cap in capitalized if len(cap) > 2 and cap.isupper() == False])
    
    return list(set(keywords))

def calculate_skills_match(cv_text: str, job_description: str) -> float:
    """Calculate skills overlap percentage"""
    common_skills = [
        'leadership', 'management', 'communication', 'teamwork', 'problem solving',
        'analytical', 'creative', 'organized', 'customer service', 'sales',
        'marketing', 'project management', 'time management', 'multitasking',
        'microsoft office', 'excel', 'powerpoint', 'google analytics',
        'social media', 'content creation', 'data analysis'
    ]
    
    cv_lower = cv_text.lower()
    job_lower = job_description.lower()
    
    # Find skills mentioned in job
    job_skills = [skill for skill in common_skills if skill in job_lower]
    
    if not job_skills:
        return 60.0
    
    # Count CV skills that match job requirements
    cv_matches = [skill for skill in job_skills if skill in cv_lower]
    
    return (len(cv_matches) / len(job_skills)) * 100

def calculate_experience_match(cv_text: str, job_description: str, job_title: str) -> float:
    """Calculate experience relevance"""
    # Look for years requirement in job
    years_pattern = r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)'
    job_years_match = re.search(years_pattern, job_description, re.IGNORECASE)
    required_years = int(job_years_match.group(1)) if job_years_match else None
    
    # Estimate CV years of experience
    cv_years = estimate_cv_experience_years(cv_text)
    
    # Calculate experience score
    if required_years and cv_years:
        if cv_years >= required_years:
            exp_score = 100
        elif cv_years >= required_years * 0.8:
            exp_score = 85
        elif cv_years >= required_years * 0.6:
            exp_score = 70
        else:
            exp_score = 50
    else:
        exp_score = 75  # Neutral when can't determine
    
    # Check for relevant job title/industry mentions
    title_words = job_title.lower().split()
    cv_lower = cv_text.lower()
    title_relevance = sum(20 for word in title_words if len(word) > 3 and word in cv_lower)
    
    return min(100, (exp_score + title_relevance) / 2)

def estimate_cv_experience_years(cv_text: str) -> int:
    """Estimate total years of experience from CV"""
    # Look for employment date ranges
    date_ranges = re.findall(r'(20\d{2}|19\d{2})\s*[-–]\s*(20\d{2}|present|current)', cv_text, re.IGNORECASE)
    
    total_years = 0
    for start_str, end_str in date_ranges:
        start_year = int(start_str)
        end_year = 2025 if end_str.lower() in ['present', 'current'] else int(end_str)
        total_years += max(0, end_year - start_year)
    
    return total_years

def calculate_semantic_similarity(cv_text: str, job_description: str) -> float:
    """Calculate semantic similarity using TF-IDF"""
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=500,
            ngram_range=(1, 2)
        )
        
        documents = [cv_text[:2000], job_description[:1000]]  # Limit length
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity * 100
        
    except Exception as e:
        logger.warning(f"Semantic similarity calculation failed: {e}")
        return 50.0

def find_missing_keywords(cv_text: str, job_description: str) -> List[str]:
    """Find job keywords missing from CV"""
    job_keywords = extract_important_keywords(job_description)
    cv_lower = cv_text.lower()
    
    missing = [kw for kw in job_keywords if kw.lower() not in cv_lower]
    return missing[:6]

def identify_strengths(cv_text: str, job_description: str) -> List[str]:
    """Identify matching strengths between CV and job"""
    job_keywords = extract_important_keywords(job_description)
    cv_lower = cv_text.lower()
    
    strengths = [kw for kw in job_keywords if kw.lower() in cv_lower]
    return strengths[:5]

def generate_improvement_areas(cv_text: str, job_description: str, keyword_score: float, skills_score: float, experience_score: float, semantic_score: float) -> List[str]:
    """Generate specific improvement areas in sentence format based on scores"""
    improvements = []
    
    # Keyword improvements
    if keyword_score < 40:
        missing_keywords = find_missing_keywords(cv_text, job_description)
        if missing_keywords:
            key_terms = ", ".join(missing_keywords[:3])
            improvements.append(f"Include more relevant industry keywords such as {key_terms} to better align with the job requirements.")
    
    # Skills improvements
    if skills_score < 50:
        improvements.append("Highlight transferable skills like project management, stakeholder collaboration, and analytical thinking that apply to this role.")
    
    # Experience improvements
    if experience_score < 40:
        improvements.append("Emphasize any experience that relates to the target industry, including volunteer work, side projects, or relevant coursework.")
    
    # Semantic/content improvements
    if semantic_score < 35:
        improvements.append("Reframe your experience using language and terminology that's more aligned with the target role and industry.")
    
    # Overall low score improvements
    overall_score = (keyword_score + skills_score + experience_score + semantic_score) / 4
    if overall_score < 30:
        improvements.append("Consider gaining relevant experience through volunteering, online courses, or networking in the target industry before applying.")
    
    # If CV is tech-heavy but job isn't
    tech_keywords = ['python', 'sql', 'api', 'cloud', 'programming', 'automation']
    if any(keyword.lower() in cv_text.lower() for keyword in tech_keywords) and not any(keyword.lower() in job_description.lower() for keyword in tech_keywords):
        improvements.append("Focus on how your technical skills can solve problems in this industry rather than listing technical tools alone.")
    
    return improvements[:4]  # Limit to 4 most important improvements

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize data and connections on startup"""
    try:
        logger.info("Starting job data initialization...")
        preload_job_embeddings()
        jobs = get_all_jobs()
        logger.info(f"Successfully loaded {len(jobs)} jobs")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Motherboard Career Assistant API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #667eea; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #667eea; }
            .method { color: #28a745; font-weight: bold; }
            .new-feature { background: linear-gradient(135deg, #f0f4ff 0%, #e8f2ff 100%); border-left: 4px solid #764ba2; }
            code { background: #e9ecef; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Motherboard Career Assistant API</h1>
            <p>Welcome to the Motherboard Career Assistant API - helping mothers navigate work and career!</p>
            
            <h2>API Status</h2>
            <p>Status: <span style="color: green;">Running</span></p>
            <p>Version: 1.0.0</p>
            
            <h2>Available Endpoints</h2>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/welcome</code>
                <p>Welcome and onboard new users</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/cv-tips</code>
                <p>Upload CV file for personalized feedback</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/search-jobs</code>
                <p>Search for jobs based on user query</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/cv-job-match</code>
                <p>Analyze CV match against specific job posting</p>
            </div>
            
            <div class="endpoint new-feature">
                <span class="method">GET</span> <code>/assessment-questions</code>
                <p>NEW! Get MomFit assessment questions and instructions</p>
            </div>
            
            <div class="endpoint new-feature">
                <span class="method">POST</span> <code>/momfit-assessment</code>
                <p>NEW! Submit MomFit assessment responses and get personalized feedback</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/assessment-stats</code>
                <p>Get assessment statistics and metadata</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/health</code>
                <p>Health check endpoint</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/docs</code>
                <p>Interactive API documentation (Swagger UI)</p>
            </div>
            
            <h2>Authentication</h2>
            <p>Most endpoints require authentication. Include your API token in the Authorization header:</p>
            <code>Authorization: Bearer YOUR_API_TOKEN</code>
            
            <h2>New Feature: MomFit Mini Assessment</h2>
            <p>Our new assessment helps working mothers understand their career readiness and work-life balance:</p>
            <ul>
                <li>20 Questions: 10 career readiness + 10 work-life balance</li>
                <li>5-7 Minutes: Quick but comprehensive evaluation</li>
                <li>Personalized Feedback: Tailored recommendations based on responses</li>
                <li>Scoring: 0-100 scale with detailed breakdowns</li>
            </ul>
            
            <h2>Africa Focus</h2>
            <p>Specialized support for working mothers in Ghana, Nigeria, and Kenya.</p>
            
            <h2>Documentation</h2>
            <p>Visit <a href="/docs">/docs</a> for interactive API documentation</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/welcome")
async def welcome_user(request: Request, token: str = Depends(verify_token)):
    """Welcome and onboard new users"""
    try:
        data = await request.json()
    except Exception:
        data = {}

    user_name = data.get("name", "there")
    returning_user = data.get("returning", False)

    if returning_user:
        welcome_message = f"Welcome back, {user_name}! How can I help you today?"
    else:
        welcome_message = (
            f"Hi {user_name}! Welcome to Motherboard - your career companion for navigating work and motherhood.\n\n"
            "I'm here to help you with:\n"
            "Job Search - Find flexible, remote, and part-time opportunities\n"
            "CV Review - Get personalized feedback on your resume\n"
            "Job Matching - Analyze how well your CV matches specific jobs\n"
            "MomFit Assessment - Understand your career readiness and work-life balance\n"
            "Africa Focus - Specialized support for Ghana, Nigeria, and Kenya\n\n"
            "What would you like to start with today?"
        )

    return JSONResponse({
        "response": welcome_message,
        "suggestions": [
            "Take the MomFit Assessment",
            "Upload my CV for review",
            "Search for jobs",
            "Analyze CV against a job posting"
        ]
    })

@app.post("/cv-tips")
async def get_cv_tips(file: UploadFile = File(...), token: str = Depends(verify_token)):
    """Get CV improvement tips"""
    try:
        if file.filename.endswith(".pdf"):
            content = extract_text_from_pdf(file.file)
        elif file.filename.endswith(".docx"):
            content = extract_text_from_docx(file.file)
        else:
            return {"error": "Unsupported file type. Please upload PDF or DOCX files."}

        messages = [
            {"role": "system", "content": "You are a professional career coach helping mothers improve their CVs. Provide specific, actionable advice focusing on how to highlight transferable skills, handle career gaps, and present experience in a compelling way for mothers returning to work or changing careers."},
            {"role": "user", "content": f"Here's my resume:\n{content}\n\nHow can I improve it?"}
        ]

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=600
        )

        return JSONResponse({
            "response": response.choices[0].message.content.strip(),
            "next_steps": [
                "Take MomFit Assessment",
                "Search for relevant jobs",
                "Analyze CV against specific job posting"
            ]
        })
    
    except Exception as e:
        logger.error(f"CV tips error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to analyze CV. Please try again."}
        )

@app.post("/search-jobs")
async def search_jobs(request: Request, token: str = Depends(verify_token)):
    """Search for jobs based on user query"""
    try:
        data = await request.json()
        user_query = data.get("query", "")
        logger.info(f"Received search query: {user_query}")
        
        if not user_query.strip():
            return JSONResponse({
                "error": "Please provide a search query",
                "suggestions": [
                    "Remote marketing jobs",
                    "Part-time healthcare positions", 
                    "Flexible teaching opportunities",
                    "Customer service work from home"
                ]
            })
        
        # Load jobs first to verify data
        jobs = get_all_jobs()
        logger.info(f"Total jobs available: {len(jobs)}")
        
        matches = find_jobs_from_sentence(user_query)
        
        # Clean matches but preserve important fields
        clean_matches = []
        for job in matches:
            job_dict = dict(job)
            # Remove only the embedding field, keep all other fields
            if 'embedding' in job_dict:
                del job_dict['embedding']
            
            # Ensure important fields are included (handle various possible column names)
            cleaned_job = {}
            
            # Handle different possible column names for company
            for company_field in ['Company', 'Company Name', 'Organization', 'Employer']:
                if company_field in job_dict:
                    cleaned_job['Company'] = job_dict[company_field]
                    break
            
            # Handle application link
            for link_field in ['Application Link or Email', 'Application Link', 'Apply Link', 'Contact Email']:
                if link_field in job_dict:
                    cleaned_job['Application Link'] = job_dict[link_field]
                    break
            
            # Handle application deadline
            for deadline_field in ['Application Deadline', 'Deadline', 'Application Due Date', 'Due Date']:
                if deadline_field in job_dict:
                    cleaned_job['Application Deadline'] = job_dict[deadline_field]
                    break
            
            # Include all other existing fields
            for key, value in job_dict.items():
                if key not in cleaned_job:  # Don't overwrite the standardized field names
                    cleaned_job[key] = value
            
            clean_matches.append(cleaned_job)
            
        logger.info(f"Found {len(clean_matches)} matches")
        
        return JSONResponse({
            "matches": clean_matches,
            "total_found": len(clean_matches),
            "search_query": user_query,
            "suggestions": [
                "Analyze CV against a specific job",
                "Get CV feedback",
                "Take MomFit Assessment"
            ] if clean_matches else [
                "Try a broader search term",
                "Search for 'remote work'",
                "Look for 'part-time jobs'"
            ]
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
    token: str = Depends(verify_token)
):
    """Hybrid CV-job match analysis: Algorithmic scoring + AI explanation"""
    try:
        # Extract CV text
        if file.filename.endswith(".pdf"):
            cv_content = extract_text_from_pdf(file.file)
        elif file.filename.endswith(".docx"):
            cv_content = extract_text_from_docx(file.file)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported file type. Please upload PDF or DOCX files."}
            )

        if not cv_content.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Could not extract text from the uploaded file."}
            )

        # STEP 1: Calculate algorithmic match scores
        match_data = calculate_cv_job_match_hybrid(cv_content, jobDescription, jobTitle)
        
        # STEP 2: Generate structured AI response
        ai_prompt = f"""
        Create a CV match analysis in this EXACT format:

        CV Match Analysis Complete!
        CV Match Score: {match_data['overall_match']}%

        Strengths:
        - [Identify 2-3 specific strengths from the CV that align with the job, written as complete sentences]

        Areas to Improve:
        - [Identify 2-3 specific areas where the CV lacks alignment with the job requirements, written as complete sentences]

        Action Steps:
        - [Provide 3 specific, actionable steps the candidate can take to improve their CV for this role]

        Pro Tip: [One encouraging insight about how their existing experience could be valuable]

        IMPORTANT INSTRUCTIONS:
        - Only mention strengths that ACTUALLY exist in the CV content provided
        - Do NOT claim the candidate has experience they don't have
        - Be truthful about gaps and mismatches
        - Focus on transferable skills, not invented experience

        CV CONTENT: {cv_content[:1500]}
        JOB: {jobTitle} at {company}
        JOB DESCRIPTION: {jobDescription[:800]}
        
        ALGORITHMIC SCORES:
        - Keyword Match: {match_data['breakdown']['keyword_match']}%
        - Skills Match: {match_data['breakdown']['skills_match']}%
        - Experience Match: {match_data['breakdown']['experience_match']}%
        
        MISSING KEYWORDS: {', '.join(match_data['missing_keywords'])}
        STRENGTHS FOUND: {', '.join(match_data['strengths'])}

        Be encouraging but completely honest about what's actually in the CV.
        """

        # Get AI explanation
        messages = [
            {
                "role": "system",
                "content": "You are a supportive career coach. Generate CV feedback in the exact format requested. Use complete sentences and be specific about improvements."
            },
            {
                "role": "user",
                "content": ai_prompt
            }
        ]

        ai_response = get_ai_response(messages, max_tokens=600)
        response_text = ai_response

        return JSONResponse({
            "response": response_text,
            "match_score": match_data['overall_match'],
            "breakdown": match_data['breakdown'],
            "strengths": match_data['strengths'],
            "missing_keywords": match_data['missing_keywords'],
            "job_title": jobTitle,
            "company": company,
            "timestamp": datetime.now().isoformat(),
            "next_steps": [
                "Apply to this job" if match_data['overall_match'] >= 70 else "Improve CV first",
                "Search for similar positions", 
                "Take MomFit Assessment",
                "Get general CV feedback"
            ]
        })

    except Exception as e:
        logger.error(f"Hybrid CV job match error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to analyze CV match. Please try again."}
        )

# NEW ASSESSMENT ENDPOINTS

@app.get("/assessment-questions")
async def get_assessment_questions(token: str = Depends(verify_token)):
    """Get assessment questions and instructions"""
    try:
        instructions = get_assessment_instructions()
        
        return JSONResponse({
            "questions": ASSESSMENT_QUESTIONS,
            "instructions": instructions,
            "metadata": {
                "total_questions": len(ASSESSMENT_QUESTIONS["career_readiness"]) + len(ASSESSMENT_QUESTIONS["work_life_balance"]),
                "career_readiness_questions": len(ASSESSMENT_QUESTIONS["career_readiness"]),
                "work_life_balance_questions": len(ASSESSMENT_QUESTIONS["work_life_balance"]),
                "estimated_time_minutes": 7,
                "scoring_scale": "1-5 (Strongly Disagree to Strongly Agree)"
            }
        })
        
    except Exception as e:
        logger.error(f"Assessment questions error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve assessment questions"}
        )

@app.post("/momfit-assessment")
async def submit_momfit_assessment(
    responses: AssessmentResponse,
    token: str = Depends(verify_token)
):
    """Submit MomFit assessment responses and get personalized feedback"""
    try:
        # Validate responses using the correct validation function
        is_valid, error_message = validate_responses(responses.dict())
        if not is_valid:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid responses", "details": error_message}
            )
        
        # Calculate scores
        scores = calculate_assessment_scores(responses.dict())
        
        # Generate personalized feedback
        feedback = generate_assessment_feedback(scores)
        
        return JSONResponse({
            "scores": {
                "career_readiness": scores["career_readiness_score"],
                "work_life_balance": scores["work_life_balance_score"], 
                "overall": scores["overall_score"]
            },
            "detailed_scores": scores,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
            "next_steps": [
                "Search for jobs that match your readiness level",
                "Get CV feedback to improve your profile",
                "Explore work-life balance resources",
                "Connect with our career coaching services"
            ]
        })
        
    except Exception as e:
        logger.error(f"MomFit assessment error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to process assessment", "details": str(e)}
        )

@app.get("/assessment-stats")
async def get_assessment_stats(token: str = Depends(verify_token)):
    """Get assessment statistics and metadata"""
    try:
        return JSONResponse({
            "assessment_info": {
                "name": "MomFit Mini Assessment",
                "version": "1.0.0",
                "total_questions": len(ASSESSMENT_QUESTIONS["career_readiness"]) + len(ASSESSMENT_QUESTIONS["work_life_balance"]),
                "categories": ["career_readiness", "work_life_balance"],
                "scoring": {
                    "scale": "1-5 Likert Scale",
                    "range": "0-100 points",
                    "interpretation": {
                        "0-40": "Needs Development",
                        "41-70": "Moderate Level", 
                        "71-85": "Strong Level",
                        "86-100": "Excellent Level"
                    }
                },
                "estimated_time": "5-7 minutes",
                "target_audience": "Working mothers and mothers seeking to return to work"
            },
            "question_breakdown": {
                "career_readiness": {
                    "count": len(ASSESSMENT_QUESTIONS["career_readiness"]),
                    "focus_areas": ["Skills confidence", "Career goals", "Professional development", "Job search readiness"]
                },
                "work_life_balance": {
                    "count": len(ASSESSMENT_QUESTIONS["work_life_balance"]),
                    "focus_areas": ["Time management", "Support systems", "Stress management", "Family-work integration"]
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Assessment stats error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve assessment statistics"}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test OpenAI connection
        test_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        openai_status = "Connected"
    except Exception:
        openai_status = "Error"
    
    try:
        # Test job data
        jobs = get_all_jobs()
        job_data_status = f"{len(jobs)} jobs loaded"
    except Exception:
        job_data_status = "Job data error"
    
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "Running",
            "openai": openai_status,
            "job_data": job_data_status,
            "assessment": "Available"
        },
        "version": "1.0.0"
    })

# Additional utility endpoints

@app.get("/api-info")
async def get_api_info():
    """Get API information and capabilities"""
    return JSONResponse({
        "name": "Motherboard Career Assistant API",
        "version": "1.0.0",
        "description": "API for helping mothers navigate work and career",
        "target_regions": ["Ghana", "Nigeria", "Kenya", "Global"],
        "capabilities": [
            "CV Analysis and Feedback",
            "Job Search and Matching",
            "CV-Job Match Analysis", 
            "MomFit Career Assessment",
            "Personalized Career Guidance"
        ],
        "supported_file_types": ["PDF", "DOCX"],
        "authentication": "Bearer Token Required",
        "rate_limits": {
            "general": "100 requests per hour",
            "file_uploads": "20 uploads per hour"
        },
        "contact": {
            "support": "support@motherboard-career.com",
            "documentation": "/docs"
        }
    })

@app.post("/feedback")
async def submit_feedback(request: Request, token: str = Depends(verify_token)):
    """Submit user feedback about the service"""
    try:
        data = await request.json()
        
        rating = data.get("rating")
        comment = data.get("comment", "")
        feature = data.get("feature", "general")
        user_id = data.get("user_id", "anonymous")
        
        # Log feedback (in production, you'd save to database)
        logger.info(f"Feedback received - Rating: {rating}, Feature: {feature}, User: {user_id}")
        logger.info(f"Comment: {comment}")
        
        return JSONResponse({
            "message": "Thank you for your feedback! We appreciate you helping us improve Motherboard.",
            "feedback_id": f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to submit feedback"}
        )

# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
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
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected error occurred",
            "message": "Please try again later or contact support",
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup message
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Motherboard Career Assistant API...")
    logger.info("Serving mothers in Ghana, Nigeria, Kenya and beyond")
    logger.info("Features: CV Analysis, Job Search, MomFit Assessment")
    uvicorn.run(app, host="0.0.0.0", port=8000)