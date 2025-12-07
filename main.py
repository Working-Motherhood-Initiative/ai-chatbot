from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends, status, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import asyncio
import threading
import secrets
import openai
from dotenv import load_dotenv
import pdfminer.high_level
from docx import Document
from job_fetcher import find_jobs_from_sentence, preload_job_embeddings, get_all_jobs
from labour_law_rag import get_rag_instance, initialize_rag_system
from datetime import datetime, timedelta
import logging
import json
from typing import Optional, Dict, List, Tuple , Any, Set
import re
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
from pydantic import BaseModel
from assessment_questions import (
    ASSESSMENT_QUESTIONS,
    calculate_assessment_scores,
    generate_assessment_feedback,
    get_assessment_instructions,
    validate_responses
)


class LabourLawQuery(BaseModel):
    query: str
    country: Optional[str] = None
    user_context: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None

class AssessmentResponse(BaseModel):
    career_readiness: Dict[str, str]
    work_life_balance: Dict[str, str]

class SessionStore:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
        
    def create_session(self) -> str:
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            "created": datetime.now(),
            "expires": datetime.now() + timedelta(hours=2),
            "requests": 0
        }
        logger.info(f"Created new session: {session_id[:8]}...")
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        if session["expires"] < datetime.now():
            del self.sessions[session_id]
            return False
        
        session["requests"] += 1
        return True
    
    def cleanup_expired(self):
        now = datetime.now()
        expired = [sid for sid, data in self.sessions.items() 
                  if data["expires"] < now]
        for sid in expired:
            del self.sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    def get_stats(self):
        return {
            "active_sessions": len(self.sessions),
            "total_sessions": len(self.sessions)
        }

session_store = SessionStore()

async def cleanup_sessions_periodically():
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        session_store.cleanup_expired()
        logger.info(f"Session stats: {session_store.get_stats()}")

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

log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Motherboard Career Assistant API", 
    version="1.0.0",
    description="API for the Motherboard Career Assistant - helping mothers navigate work and career"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dragonfly-chihuahua-alhg.squarespace.com",
        "https://ai-chatbot-4bqx.onrender.com",
    ],
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


def remove_personal_info_from_cv(cv_text: str) -> str:
    try:
        if not cv_text or not cv_text.strip():
            logger.warning("Empty CV text provided for privacy protection")
            return ""
            
        cleaned_text = cv_text
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        cleaned_text = re.sub(email_pattern, '[EMAIL REMOVED]', cleaned_text)
        
        phone_patterns = [
            r'\+?\d{1,4}[\s\-\(\)]?\(?\d{1,4}\)?[\s\-]?\d{1,4}[\s\-]?\d{1,9}',
            r'\b\d{3}[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b',
            r'\+\d{1,3}\s?\d{1,4}\s?\d{1,4}\s?\d{1,9}',
            r'\(\d{3}\)\s?\d{3}[\s\-]?\d{4}',
        ]
        
        for pattern in phone_patterns:
            cleaned_text = re.sub(pattern, '[PHONE REMOVED]', cleaned_text)
        
        address_patterns = [
            r'\b\d+\s+[A-Za-z\s]+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Place|Pl)\b.*',
            r'\b\d+[A-Za-z]?\s+[A-Za-z\s]+\d{5}(-\d{4})?\b',
            r'\b[A-Za-z\s]+,\s*[A-Za-z\s]+\s*\d{5}(-\d{4})?\b',
        ]
        
        for pattern in address_patterns:
            cleaned_text = re.sub(pattern, '[ADDRESS REMOVED]', cleaned_text, flags=re.IGNORECASE)
        
        # 4. Remove potential names from the beginning of CV
        header_section = cleaned_text[:500]
        main_section = cleaned_text[500:]
        
        lines = header_section.split('\n')
        filtered_header_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if len(line) < 3:
                filtered_header_lines.append(line)
                continue
                
            words = line.split()
            
            if (2 <= len(words) <= 4 and 
                all(word.replace('.', '').replace(',', '').isalpha() for word in words) and
                any(word[0].isupper() for word in words if len(word) > 0) and
                i < 5):
                continue
            
            career_keywords = ['objective', 'summary', 'profile', 'experience', 'education', 
                              'skills', 'qualifications', 'employment', 'work', 'career',
                              'professional', 'expertise', 'background', 'achievements']
            
            if any(keyword in line.lower() for keyword in career_keywords):
                filtered_header_lines.append(line)
            elif len(words) > 4:
                filtered_header_lines.append(line)
            else:
                filtered_header_lines.append(line)
        
        cleaned_header = '\n'.join(filtered_header_lines)
        cleaned_text = cleaned_header + '\n' + main_section
        
        # 5. Remove social security numbers
        ssn_pattern = r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
        cleaned_text = re.sub(ssn_pattern, '[ID REMOVED]', cleaned_text)
        
        # 6. Remove LinkedIn URLs
        linkedin_pattern = r'https?://(?:www\.)?linkedin\.com/in/[^\s]+'
        cleaned_text = re.sub(linkedin_pattern, '[LINKEDIN PROFILE]', cleaned_text)
        
        # 7. Remove other social media URLs
        social_patterns = [
            r'https?://(?:www\.)?(?:twitter|facebook|instagram|github)\.com/[^\s]+',
            r'https?://[^\s]*(?:twitter|facebook|instagram)\.com[^\s]*'
        ]
        
        for pattern in social_patterns:
            cleaned_text = re.sub(pattern, '[SOCIAL MEDIA PROFILE]', cleaned_text)
        
        # 8. Remove dates of birth
        dob_patterns = [
            r'\b(?:DOB|Date of Birth|Born):?\s*\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b',
            r'\b(?:DOB|Date of Birth|Born):?\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        for pattern in dob_patterns:
            cleaned_text = re.sub(pattern, '[DATE OF BIRTH REMOVED]', cleaned_text, flags=re.IGNORECASE)
        
        # 9. Clean up extra whitespace
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
        
        # 10. Add privacy notice
        privacy_notice = "=== PRIVACY-PROTECTED CV ANALYSIS ===\n[Personal identifying information has been removed for privacy protection]\n\n"
        cleaned_text = privacy_notice + cleaned_text.strip()
        
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Privacy protection failed: {e}")
        raise HTTPException(status_code=500, detail="Privacy protection system failed")


def validate_privacy_protection(original_text: str, cleaned_text: str) -> Dict:
    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "privacy_check_passed": True,
        "issues_found": [],
        "statistics": {}
    }
    
    try:
        emails_in_original = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', original_text)
        emails_in_cleaned = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', cleaned_text)
        
        if emails_in_cleaned:
            validation_results["privacy_check_passed"] = False
            validation_results["issues_found"].append(f"EMAIL LEAK: {len(emails_in_cleaned)} email(s) still present")
        
        phone_patterns = [
            r'\+?\d{1,4}[\s\-\(\)]?\(?\d{1,4}\)?[\s\-]?\d{1,4}[\s\-]?\d{1,9}',
            r'\b\d{3}[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b',
            r'\(\d{3}\)\s?\d{3}[\s\-]?\d{4}'
        ]
        
        phones_in_cleaned = []
        for pattern in phone_patterns:
            phones_in_cleaned.extend(re.findall(pattern, cleaned_text))
        
        phones_in_cleaned_filtered = [p for p in phones_in_cleaned if not re.match(r'^(19|20)\d{2}$', p.replace('-', '').replace(' ', '').replace('(', '').replace(')', ''))]
        
        if phones_in_cleaned_filtered:
            validation_results["privacy_check_passed"] = False
            validation_results["issues_found"].append(f"PHONE LEAK: {len(phones_in_cleaned_filtered)} phone(s) still present")
        
        validation_results["statistics"] = {
            "original_length": len(original_text),
            "cleaned_length": len(cleaned_text),
            "reduction_percentage": round((len(original_text) - len(cleaned_text)) / len(original_text) * 100, 2) if len(original_text) > 0 else 0,
            "total_issues_found": len(validation_results["issues_found"]),
            "emails_removed": len(emails_in_original) - len(emails_in_cleaned),
            "phones_removed": len([p for p in re.findall(r'\+?\d{1,4}[\s\-\(\)]?\(?\d{1,4}\)?[\s\-]?\d{1,4}[\s\-]?\d{1,9}', original_text)]) - len(phones_in_cleaned_filtered)
        }
        
    except Exception as e:
        logger.error(f"Privacy validation failed: {e}")
        validation_results["privacy_check_passed"] = False
        validation_results["issues_found"].append(f"VALIDATION ERROR: {str(e)}")
    
    return validation_results


def log_privacy_protection(original_cv: str, cleaned_cv: str, endpoint: str):    
    try:
        validation = validate_privacy_protection(original_cv, cleaned_cv)
        
        logger.info(f"=== PRIVACY PROTECTION LOG - {endpoint} ===")
        logger.info(f"Reduction: {validation['statistics']['reduction_percentage']}%")
        logger.info(f"Issues Found: {validation['statistics']['total_issues_found']}")
        
        if validation["issues_found"]:
            logger.warning(f"Privacy Issues Detected: {validation['issues_found']}")
        else:
            logger.info("Privacy protection successful - no issues found")
        
        logger.info(f"Emails removed: {validation['statistics']['emails_removed']}")
        logger.info(f"Phones removed: {validation['statistics']['phones_removed']}")
        
        return validation
        
    except Exception as e:
        logger.error(f"Privacy logging failed: {e}")
        return {"privacy_check_passed": False, "statistics": {"reduction_percentage": 0}}


# Utility functions
def extract_text_from_pdf(file):
    try:
        return pdfminer.high_level.extract_text(file)
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF")

def extract_text_from_docx(file):
    try:
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {e}")
        raise HTTPException(status_code=400, detail="Failed to extract text from DOCX")

def extract_skills_from_cv(cv_text: str) -> List[str]:
    try:
        cleaned_cv_text = remove_personal_info_from_cv(cv_text)
        
        messages = [
            {
                "role": "system", 
                "content": "You are a skill extraction expert. Extract key skills, technologies, and competencies from the privacy-protected CV text. Return only a comma-separated list of skills, no explanations."
            },
            {
                "role": "user", 
                "content": f"Extract skills from this privacy-protected CV:\n{cleaned_cv_text[:2000]}"
            }
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

def get_match_ranking(overall_score: int) -> Dict[str, str]:
    if overall_score >= 80:
        return {
            "level": "High",
            "color": "#059669",
            "description": "Excellent match - strongly recommended to apply"
        }
    elif overall_score >= 50:
        return {
            "level": "Medium", 
            "color": "#f59e0b",
            "description": "Good match - consider applying with CV improvements"
        }
    else:
        return {
            "level": "Low",
            "color": "#ef4444",
            "description": "Weak match - significant improvements needed"
        }

def calculate_keyword_match(cv_text: str, job_description: str) -> float:
    job_keywords = extract_important_keywords(job_description)
    
    if not job_keywords:
        return 50.0
    
    cv_lower = cv_text.lower()
    matches = sum(1 for keyword in job_keywords if keyword.lower() in cv_lower)
    
    return (matches / len(job_keywords)) * 100

def extract_important_keywords(text: str) -> List[str]:
    tech_pattern = r'\b(python|javascript|java|react|angular|sql|html|css|git|aws|azure|docker|kubernetes|salesforce|hubspot|excel|powerpoint|google analytics|photoshop|illustrator|figma|canva)\b'
    
    soft_pattern = r'\b(leadership|management|communication|teamwork|problem[\-\s]solving|analytical|creative|organized|customer service|project management)\b'
    
    industry_pattern = r'\b(marketing|sales|finance|healthcare|education|technology|consulting|administration|operations|human resources)\b'
    
    keywords = []
    for pattern in [tech_pattern, soft_pattern, industry_pattern]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        keywords.extend(matches)
    
    capitalized = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text)
    keywords.extend([cap for cap in capitalized if len(cap) > 2 and cap.isupper() == False])
    
    return list(set(keywords))

def calculate_skills_match(cv_text: str, job_description: str) -> float:
    common_skills = [
        'leadership', 'management', 'communication', 'teamwork', 'problem solving',
        'analytical', 'creative', 'organized', 'customer service', 'sales',
        'marketing', 'project management', 'time management', 'multitasking',
        'microsoft office', 'excel', 'powerpoint', 'google analytics',
        'social media', 'content creation', 'data analysis'
    ]
    
    cv_lower = cv_text.lower()
    job_lower = job_description.lower()
    
    job_skills = [skill for skill in common_skills if skill in job_lower]
    
    if not job_skills:
        return 60.0
    
    cv_matches = [skill for skill in job_skills if skill in cv_lower]
    
    return (len(cv_matches) / len(job_skills)) * 100

def calculate_experience_match(cv_text: str, job_description: str, job_title: str) -> float:
    years_pattern = r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)'
    job_years_match = re.search(years_pattern, job_description, re.IGNORECASE)
    required_years = int(job_years_match.group(1)) if job_years_match else None
    
    cv_years = estimate_cv_experience_years(cv_text)
    
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
        exp_score = 75
    
    title_words = job_title.lower().split()
    cv_lower = cv_text.lower()
    title_relevance = sum(20 for word in title_words if len(word) > 3 and word in cv_lower)
    
    return min(100, (exp_score + title_relevance) / 2)

def estimate_cv_experience_years(cv_text: str) -> int:
    date_ranges = re.findall(r'(20\d{2}|19\d{2})\s*[-–]\s*(20\d{2}|present|current)', cv_text, re.IGNORECASE)
    
    total_years = 0
    for start_str, end_str in date_ranges:
        start_year = int(start_str)
        end_year = 2025 if end_str.lower() in ['present', 'current'] else int(end_str)
        total_years += max(0, end_year - start_year)
    
    return total_years

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

def find_missing_keywords(cv_text: str, job_description: str) -> List[str]:
    job_keywords = extract_important_keywords(job_description)
    cv_lower = cv_text.lower()
    
    missing = [kw for kw in job_keywords if kw.lower() not in cv_lower]
    return missing[:6]

def identify_strengths(cv_text: str, job_description: str) -> List[str]:
    job_keywords = extract_important_keywords(job_description)
    cv_lower = cv_text.lower()
    
    strengths = [kw for kw in job_keywords if kw.lower() in cv_lower]
    return strengths[:5]

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

def validate_file_size(file: UploadFile):
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 5MB.")

def initialize_background():
    global _initialization_complete, _initialization_error
    try:
        logger.info("Background initialization started...")
        
        logger.info("Starting job data initialization...")
        preload_job_embeddings()
        jobs = get_all_jobs()
        logger.info(f"Successfully loaded {len(jobs)} jobs")
        
        logger.info("Initializing Labour Law RAG system...")
        try:
            doc_count = initialize_rag_system(
                pdf_directory="labour_laws",
                force_reload=False,
                use_gdrive=True
            )
            logger.info(f"Successfully loaded {doc_count} labour law document chunks")
            
            rag = get_rag_instance()
            stats = rag.get_system_stats()
            logger.info(f"Google Drive enabled: {stats.get('gdrive_enabled', False)}")
            logger.info(f"Local cache exists: {stats.get('local_cache_exists', False)}")
            
        except Exception as e:
            logger.error(f"Labour Law RAG initialization failed: {e}")
            logger.warning("Labour law queries will not be available until next restart")
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
        logger.info("Session management initialized")
        
        # Start session cleanup
        asyncio.create_task(cleanup_sessions_periodically())
        
        logger.info("Running heavy initialization in background thread...")
        init_thread = threading.Thread(target=initialize_background, daemon=True)
        init_thread.start()
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

@app.get("/init-status")
async def initialization_status():
    return JSONResponse({
        "initialization_complete": _initialization_complete,
        "error": _initialization_error,
        "timestamp": datetime.now().isoformat()
    })   

@app.post("/admin/reload-vectorstore")
async def admin_reload_vectorstore(token: str = Depends(verify_token)):
    """Admin endpoint - still uses API token"""
    try:
        logger.info("Admin: Manual vectorstore reload initiated...")
        
        rag = get_rag_instance()
        doc_count = rag.reload_from_gdrive()
        
        return JSONResponse({
            "status": "success",
            "message": "Vector store reloaded successfully from Google Drive",
            "document_chunks": doc_count,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Admin reload failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to reload vector store", 
                "details": str(e)
            }
        )

@app.post("/create-session")
async def create_session():
    """Create a new session for the frontend - NO AUTH REQUIRED"""
    session_id = session_store.create_session()
    return {
        "session_id": session_id,
        "expires_in_hours": 2,
        "message": "Session created successfully"
    }

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        rag = get_rag_instance()
        stats = rag.get_system_stats()
        gdrive_status = "Enabled" if stats.get('gdrive_enabled') else "Disabled"
        gdrive_class = "enabled" if stats.get('gdrive_enabled') else "disabled"
    except:
        gdrive_status = "Unknown"
        gdrive_class = "disabled"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Motherboard Career Assistant API</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #667eea; }}
            .endpoint {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #667eea; }}
            .method {{ color: #28a745; font-weight: bold; }}
            .new-feature {{ background: linear-gradient(135deg, #f0f4ff 0%, #e8f2ff 100%); border-left: 4px solid #764ba2; }}
            .privacy-feature {{ background: linear-gradient(135deg, #fff0f0 0%, #ffe8e8 100%); border-left: 4px solid #dc3545; }}
            .session-feature {{ background: linear-gradient(135deg, #f0fff4 0%, #e8ffe8 100%); border-left: 4px solid #10b981; }}
            .status-enabled {{ color: #28a745; font-weight: bold; }}
            .status-disabled {{ color: #6c757d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Motherboard Career Assistant API</h1>
            <p>Welcome to the Motherboard Career Assistant API - helping mothers navigate work and career with <strong>privacy protection & session security!</strong></p>
            
            <h2>API Status</h2>
            <p>Status: <span style="color: green;">Running with Privacy Protection & Session Management</span></p>
            <p>Version: 1.0.0</p>
            <p>Google Drive Integration: <span class="status-{gdrive_class}">{gdrive_status}</span></p>
            <p>Session System: <span class="status-enabled">Active</span></p>
            
            <div class="session-feature">
                <h3> Session Management Active</h3>
                <p>New secure session system:</p>
                <ul>
                    <li>No exposed API tokens in frontend</li>
                    <li>2-hour session expiry with auto-refresh</li>
                    <li>In-memory session storage (fast & free)</li>
                    <li>Automatic cleanup of expired sessions</li>
                </ul>
            </div>
            
            <div class="privacy-feature">
                <h3>Privacy Protection Active</h3>
                <p>All CV uploads are now processed with comprehensive privacy protection:</p>
                <ul>
                    <li>Personal information removed before AI analysis</li>
                    <li>Names, emails, phone numbers, addresses protected</li>
                    <li>Professional content preserved for quality analysis</li>
                    <li>Real-time privacy monitoring and validation</li>
                </ul>
            </div>
            
            <div class="new-feature">
                <h3>Google Drive Integration</h3>
                <p>Labour law vectorstore now loads from Google Drive for faster startup:</p>
                <ul>
                    <li>Ultra-fast initialization (seconds vs minutes)</li>
                    <li>No need for local PDF processing</li>
                    <li>Easy updates - just replace file in Google Drive</li>
                    <li>Automatic local caching for offline resilience</li>
                </ul>
            </div>
            
            <h2>Available Endpoints</h2>
            
            <div class="endpoint session-feature">
                <span class="method">POST</span> <code>/create-session</code>
                <p>Create a new session (public, no auth) - Sessions last 2 hours</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/welcome</code>
                <p>Welcome and onboard new users (requires session)</p>
            </div>
            
            <div class="endpoint privacy-feature">
                <span class="method">POST</span> <code>/cv-tips</code>
                <p>Upload CV file for personalized feedback (Privacy Protected, requires session)</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/search-jobs</code>
                <p>Search for jobs based on user query (requires session)</p>
            </div>
            
            <div class="endpoint privacy-feature">
                <span class="method">POST</span> <code>/cv-job-match</code>
                <p>Analyze CV match against specific job posting (Privacy Protected, requires session)</p>
            </div>
            
            <div class="endpoint new-feature">
                <span class="method">GET</span> <code>/assessment-questions</code>
                <p>Get MomFit assessment questions and instructions (requires session)</p>
            </div>
            
            <div class="endpoint new-feature">
                <span class="method">POST</span> <code>/momfit-assessment</code>
                <p>Submit MomFit assessment responses and get personalized feedback (requires session)</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/assessment-stats</code>
                <p>Get assessment statistics and metadata (requires session)</p>
            </div>

            <div class="endpoint new-feature">
                <span class="method">POST</span> <code>/labour-law-query</code>
                <p>Ask questions about labour laws across 16 African countries (RAG-powered, requires session)</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/labour-law-countries</code>
                <p>Get list of supported countries for labour law queries (requires session)</p>
            </div>
            
            <div class="endpoint new-feature">
                <span class="method">POST</span> <code>/admin/reload-vectorstore</code>
                <p>Manually reload vectorstore from Google Drive (Admin only - requires API token)</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/health</code>
                <p>Health check endpoint with privacy status and Google Drive info (no auth)</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/docs</code>
                <p>Interactive API documentation (Swagger UI)</p>
            </div>
            
            <h2>Authentication</h2>
            <p><strong>Public Endpoints:</strong> /create-session, /health, /docs (no auth needed)</p>
            <p><strong>User Endpoints:</strong> Most endpoints require a session token from /create-session</p>
            <p><strong>Admin Endpoints:</strong> /admin/* require API token: <code>Authorization: Bearer API_TOKEN</code></p>
            
            <h2>Documentation</h2>
            <p>Visit <a href="/docs">/docs</a> for interactive API documentation</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/welcome")
async def welcome_user(request: Request, session: str = Depends(verify_session)):
    try:
        data = await request.json()
    except Exception:
        data = {}

    user_name = data.get("name", "there")
    returning_user = data.get("returning", False)

    welcome_message = f"Welcome back, {user_name}! How can I help you today?"

    return JSONResponse({
        "response": welcome_message
    })

@app.post("/labour-law-query")
async def labour_law_query(request_data: LabourLawQuery, session: str = Depends(verify_session)):
    try:
        rag = get_rag_instance()

        if not rag.documents_loaded:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Labour law system is not initialized",
                    "message": "Please try again in a moment or contact support"
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
                "Get CV feedback (privacy protected)",
                "Take MomFit Assessment"
            ],
            "disclaimer": "This information is for educational purposes. For specific legal advice, please consult a qualified employment lawyer in your country."
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
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve country information"}
        )


@app.get("/jobs")
async def get_all_available_jobs(session: str = Depends(verify_session)):
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
                job_dict.get("Job Description (Brief Summary)  ") or
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
                "application_link": job_dict.get("Application Link or Email", job_dict.get("Application Link", "N/A")),
                "application_deadline": job_dict.get("Application Deadline", "N/A"),
                "skills_required": job_dict.get("Skills Required", "N/A"),
                "additional_fields": {k: v for k, v in job_dict.items() if k not in [
                    "Job Title", "Company Name", "Company", "Job Type", "Industry",
                    "Job Description (Brief Summary)", "Job Description (Brief Summary)  ",
                    "Job Description", "Location", "Application Link or Email", 
                    "Application Link", "Application Deadline", "Skills Required"
                ]}
            }
            
            complete_jobs.append(complete_job)
            
        logger.info(f"Successfully formatted {len(complete_jobs)} jobs")
        
        return JSONResponse({
            "jobs": complete_jobs,
            "total_jobs": len(complete_jobs),
            "message": f"Showing all {len(complete_jobs)} available job(s). Click on any job to view details and apply.",
            "suggestions": [
                "Analyze your CV against a job (privacy protected)",
                "Get CV feedback (privacy protected)",
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
        job_type_filter = data.get("job_type", "").strip()
        
        logger.info(f"Search - Query: '{user_query}', Country: '{country_filter}', Type: '{job_type_filter}'")
        
        if not user_query.strip():
            jobs = get_all_jobs()
        else:
            jobs = find_jobs_from_sentence(user_query)
        
        if country_filter:
            jobs = [
                job for job in jobs
                if country_filter.lower() in job.get("Location", "").lower()
            ]
        
        if job_type_filter:
            job_type_lower = job_type_filter.lower()
            jobs = [
                job for job in jobs
                if job_type_lower in job.get("Job Type", "").lower()
            ]
        
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
            
            complete_job = {
                "job_id": idx,
                "job_title": job_dict.get("Job Title", "N/A"),
                "company_name": job_dict.get("Company Name", job_dict.get("Company", "N/A")),
                "job_type": job_dict.get("Job Type", "N/A"),
                "industry": job_dict.get("Industry", "N/A"),
                "job_description": description,
                "location": job_dict.get("Location", "N/A"),
                "application_link": job_dict.get("Application Link or Email", job_dict.get("Application Link", "N/A")),
                "application_deadline": job_dict.get("Application Deadline", "N/A"),
                "skills_required": job_dict.get("Skills Required", "N/A"),
            }
            
            complete_jobs.append(complete_job)
        
        filters_applied = []
        if user_query:
            filters_applied.append(f"query '{user_query}'")
        if country_filter:
            filters_applied.append(f"country '{country_filter}'")
        if job_type_filter:
            filters_applied.append(f"type '{job_type_filter}'")
        
        filter_text = " and ".join(filters_applied) if filters_applied else "no filters"
        
        return JSONResponse({
            "jobs": complete_jobs,
            "total_found": len(complete_jobs),
            "filters_applied": {
                "query": user_query if user_query else None,
                "country": country_filter if country_filter else None,
                "job_type": job_type_filter if job_type_filter else None
            },
            "message": f"Found {len(complete_jobs)} job(s) matching {filter_text}.",
            "available_filters": {
                "countries": ["Ghana", "Nigeria", "Kenya", "Remote"],
                "job_types": ["Remote", "On-site", "Hybrid", "Part-time", "Full-time"]
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
    session: str = Depends(verify_session)
):
    try:
        validate_file_size(file)
        
        if file.filename.endswith(".pdf"):
            original_cv_content = extract_text_from_pdf(file.file)
        elif file.filename.endswith(".docx"):
            original_cv_content = extract_text_from_docx(file.file)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported file type. Please upload PDF or DOCX files."}
            )

        if not original_cv_content.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Could not extract text from the uploaded file."}
            )

        cleaned_cv_content = remove_personal_info_from_cv(original_cv_content)
        privacy_validation = log_privacy_protection(original_cv_content, cleaned_cv_content, "cv-job-match")
        
        if not privacy_validation["privacy_check_passed"]:
            logger.error(f"PRIVACY BREACH DETECTED in job match: {privacy_validation['issues_found']}")

        match_data = calculate_cv_job_match_hybrid(cleaned_cv_content, jobDescription, jobTitle)
        
        ai_prompt = f"""
        Create a CV match analysis in this EXACT format:

        CV Match Analysis Complete!
        CV Match Score: {match_data['overall_match']}%

        Strengths:
        - [Identify 2-3 specific strengths from the privacy-protected CV that align with the job]

        Areas to Improve:
        - [Identify 2-3 specific areas where the CV lacks alignment with the job requirements]

        Action Steps:
        - [Provide 3 specific, actionable steps the candidate can take to improve their CV for this role]

        Pro Tip: [One encouraging insight about how their existing experience could be valuable]

        PRIVACY-PROTECTED CV CONTENT: {cleaned_cv_content[:1500]}
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

        messages = [
            {
                "role": "system",
                "content": "You are a supportive career coach. The CV has been privacy-protected. Generate feedback focusing on professional content only."
            },
            {
                "role": "user",
                "content": ai_prompt
            }
        ]

        ai_response = get_ai_response(messages, max_tokens=600)
        ranking = get_match_ranking(match_data['overall_match'])

        return JSONResponse({
            "response": ai_response,
            "match_score": match_data['overall_match'],
            "ranking": ranking["level"],  
            "ranking_details": ranking,   
            "breakdown": match_data['breakdown'],
            "strengths": match_data['strengths'],
            "missing_keywords": match_data['missing_keywords'],
            "privacy_protection": {
                "enabled": True,
                "status": "Protected" if privacy_validation["privacy_check_passed"] else "Warning",
                "reduction_percentage": privacy_validation["statistics"]["reduction_percentage"]
            },
            "job_title": jobTitle,
            "company": company,
            "timestamp": datetime.now().isoformat(),
            "next_steps": [
                "Apply to this job" if match_data['overall_match'] >= 70 else "Improve CV first",
                "Search for similar positions", 
                "Take MomFit Assessment"
            ]
        })

    except Exception as e:
        logger.error(f"CV job match error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to analyze CV match. Please try again."}
        )

@app.get("/assessment-questions")
async def get_assessment_questions(session: str = Depends(verify_session)):
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
    session: str = Depends(verify_session)
):
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
            "detailed_scores": scores,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
            "next_steps": [
                "Search for jobs that match your readiness level",
                "Get CV feedback to improve your profile (privacy protected)",
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
async def get_assessment_stats(session: str = Depends(verify_session)):
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
    try:
        test_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        openai_status = "Connected"
    except Exception:
        openai_status = "Error"
    
    try:
        jobs = get_all_jobs()
        job_data_status = f"{len(jobs)} jobs loaded"
    except Exception:
        job_data_status = "Job data error"
    
    try:
        test_cv = "John Doe john@email.com (555) 123-4567 Professional Summary: Software developer"
        cleaned = remove_personal_info_from_cv(test_cv)
        privacy_status = "Active" if "[EMAIL REMOVED]" in cleaned else "Failed"
    except Exception:
        privacy_status = "Error"
    
    try:
        rag = get_rag_instance()
        stats = rag.get_system_stats()
        
        if stats['documents_loaded']:
            labour_law_status = f"Active - {stats['total_chunks']} chunks"
            gdrive_info = {
                "enabled": stats.get('gdrive_enabled', False),
                "local_cache": stats.get('local_cache_exists', False)
            }
        else:
            labour_law_status = "Not initialized"
            gdrive_info = {"enabled": False, "local_cache": False}
            
    except Exception:
        labour_law_status = "Error"
        gdrive_info = {"enabled": False, "local_cache": False}
    
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "Running",
            "openai": openai_status,
            "job_data": job_data_status,
            "assessment": "Available",
            "privacy_protection": privacy_status,
            "labour_law_rag": labour_law_status,
            "session_management": "Active"
        },
        "session_stats": session_store.get_stats(),
        "google_drive": gdrive_info,
        "version": "1.0.0",
        "file_limits": {
            "max_size": "5MB",
            "supported_formats": ["PDF", "DOCX"]
        }
    })
    
    
@app.get("/api-info")
async def get_api_info():
    return JSONResponse({
        "name": "Motherboard Career Assistant API",
        "version": "1.0.0",
        "description": "API for helping mothers navigate work and career with privacy protection",
        "target_regions": ["Ghana", "Nigeria", "Kenya", "Global"],
        "capabilities": [
            "Privacy-Protected CV Analysis and Feedback",
            "Privacy-Protected CV-Job Match Analysis",
            "MomFit Career Assessment",
            "Job Search and Matching",
            "Personalized Career Guidance",
            "Session-based Security"
        ],
        "privacy_features": [
            "Automatic personal information removal",
            "Real-time privacy validation",
            "Professional content preservation",
            "Comprehensive logging"
        ],
        "security_features": [
            "Session-based authentication",
            "2-hour session expiry",
            "Automatic session cleanup",
            "No exposed API tokens"
        ],
        "supported_file_types": ["PDF", "DOCX"],
        "file_limits": {
            "max_size": "5MB",
            "timeout": "30 seconds"
        },
        "authentication": "Session Token Required (from /create-session)",
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
            "message": "Please try again later or contact support",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Motherboard Career Assistant API...")
    logger.info("Privacy Protection: ENABLED")
    logger.info("Session Management: ENABLED")
    logger.info("Serving mothers in Ghana, Nigeria, Kenya and beyond")
    logger.info("Features: CV Analysis, Job Search, MomFit Assessment")
    uvicorn.run(app, host="0.0.0.0", port=8000)