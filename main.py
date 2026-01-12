from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends, status, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import asyncio
import logging
from datetime import datetime
from typing import Optional
import httpx
import json
import hmac
import hashlib
from sqlalchemy import create_engine, Column, String, Boolean, DateTime, Integer, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
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


DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
elif not DATABASE_URL.startswith("postgresql+psycopg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    paystack_customer_code = Column(String)
    paystack_customer_id = Column(Integer)
    authorization_code = Column(String, nullable=True)
    email_token = Column(String, nullable=True)           
    first_authorization = Column(Boolean, default=False)
    subscription_active = Column(Boolean, default=False)
    subscription_code = Column(String, nullable=True)
    last_payment_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True)
    subscription_code = Column(String, unique=True, index=True)
    plan_id = Column(String)
    status = Column(String)
    email_token = Column(String, nullable=True)          
    next_payment_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PaymentLog(Base):
    __tablename__ = "payment_logs"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True)
    reference = Column(String, unique=True, index=True)
    amount = Column(Integer)
    status = Column(String)
    event_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app = FastAPI(
    title="Motherboard Career Assistant API", 
    version="2.0.0",
    description="Complete API for Motherboard - Career Guidance + Payment Management"
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

PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY")
PAYSTACK_PUBLIC_KEY = os.getenv("PAYSTACK_PUBLIC_KEY")
PAYSTACK_BASE_URL = "https://api.paystack.co"

paystack_headers = {
    "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
    "Content-Type": "application/json"
}


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


def validate_email(email: str) -> str:
    if not email or '@' not in email or '.' not in email:
        raise ValueError('Invalid email format')
    return email.lower().strip()


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
async def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected", "version": "2.0.0"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

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

#PAYMENT API ENDPOINTS

@app.post("/api/customers")
async def create_customer(request: Request, db: Session = Depends(get_db)):
    try:
        data = await request.json()
        email = data.get('email', '').strip()
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()

        if not email or not first_name or not last_name:
            raise HTTPException(status_code=400, detail="email, first_name, and last_name are required")

        validate_email(email)

        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Customer already exists")

        async with httpx.AsyncClient() as client:
            payload = {
                "email": email,
                "first_name": first_name,
                "last_name": last_name
            }

            response = await client.post(
                f"{PAYSTACK_BASE_URL}/customer",
                json=payload,
                headers=paystack_headers,
                timeout=30.0
            )

            if response.status_code not in (200, 201):
                try:
                    err = response.json()
                except Exception:
                    err = response.text
                raise HTTPException(status_code=400, detail=f"Failed to create customer on Paystack: {err}")

            paystack_customer = response.json().get("data")

            db_user = User(
                email=email,
                first_name=first_name,
                last_name=last_name,
                paystack_customer_code=paystack_customer.get("customer_code"),
                paystack_customer_id=paystack_customer.get("id"),
                subscription_active=False
            )
            db.add(db_user)
            db.commit()
            db.refresh(db_user)

            return {
                "status": "success",
                "message": "Customer created successfully",
                "data": {
                    "email": email,
                    "customer_code": paystack_customer.get("customer_code")
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/initialize-payment")
async def initialize_payment(request: Request, db: Session = Depends(get_db)):
    try:
        data = await request.json()
        email = data.get('email', '').strip()

        if not email:
            raise HTTPException(status_code=400, detail="email is required")

        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(status_code=404, detail="Customer not found. Create customer first.")

        async with httpx.AsyncClient() as client:
            payload = {
                "email": email,
                "amount": 8000,
                "channels": ["card", "bank", "ussd", "qr", "mobile_money"],
                "metadata": {
                    "plan_name": "Motherboard Monthly Plan",
                    "subscription_type": "monthly"
                }
            }

            response = await client.post(
                f"{PAYSTACK_BASE_URL}/transaction/initialize",
                json=payload,
                headers=paystack_headers,
                timeout=30.0
            )

            if response.status_code not in (200, 201):
                raise HTTPException(status_code=400, detail="Failed to initialize payment")

            transaction_data = response.json().get("data")

            return {
                "status": "success",
                "message": "Payment initialized",
                "data": {
                    "authorization_url": transaction_data.get("authorization_url"),
                    "access_code": transaction_data.get("access_code"),
                    "reference": transaction_data.get("reference")
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/verify-payment")
async def verify_payment(request: Request, db: Session = Depends(get_db)):
    try:
        data = await request.json()
        reference = data.get('reference', '').strip()

        if not reference:
            raise HTTPException(status_code=400, detail="reference is required")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{PAYSTACK_BASE_URL}/transaction/verify/{reference}",
                headers=paystack_headers,
                timeout=30.0
            )

            if response.status_code not in (200, 201):
                raise HTTPException(status_code=400, detail="Failed to verify payment")

            transaction = response.json().get("data")

            if transaction.get("status") != "success":
                return {
                    "status": "failed",
                    "message": "Payment was not successful"
                }

            customer_email = transaction["customer"]["email"]
            authorization_data = transaction.get("authorization") or {}
            amount = transaction.get("amount")

            user = db.query(User).filter(User.email == customer_email).first()
            if user:
                auth_code = authorization_data.get("authorization_code")
                if auth_code:
                    user.authorization_code = auth_code
                    user.first_authorization = True
                    user.updated_at = datetime.utcnow()
                    db.commit()

            payment_log = PaymentLog(
                email=customer_email,
                reference=reference,
                amount=amount,
                status="success",
                event_type="initial_payment",
                metadata_json=json.dumps(authorization_data)
            )
            db.add(payment_log)
            db.commit()

            return {
                "status": "success",
                "message": "Payment verified!",
                "data": {
                    "email": customer_email,
                    "amount": amount / 100 if amount else None,
                    "reference": reference
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/create-subscription")
async def create_subscription(request: Request, db: Session = Depends(get_db)):
    try:
        data = await request.json()
        email = data.get('email', '').strip()

        if not email:
            raise HTTPException(status_code=400, detail="email is required")

        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(status_code=404, detail="Customer not found")

        if not user.authorization_code:
            raise HTTPException(status_code=400, detail="Customer must complete initial payment first")

        async with httpx.AsyncClient() as client:
            payload = {
                "customer": user.paystack_customer_code,
                "plan": "PLN_u6si72zqqto8dq0",
                "authorization": user.authorization_code,
                "start_date": datetime.utcnow().isoformat()
            }

            response = await client.post(
                f"{PAYSTACK_BASE_URL}/subscription",
                json=payload,
                headers=paystack_headers,
                timeout=30.0
            )

            if response.status_code not in (200, 201):
                try:
                    error_msg = response.json().get("message", "Failed to create subscription")
                except Exception:
                    error_msg = response.text or "Failed to create subscription"
                raise HTTPException(status_code=400, detail=error_msg)

            body = response.json()
            if not body.get("status"):
                raise HTTPException(status_code=400, detail=body.get("message", "Paystack error"))

            subscription_data = body.get("data") or {}

            db_subscription = Subscription(
                email=email,
                subscription_code=subscription_data.get("subscription_code"),
                plan_id=subscription_data.get("plan"),
                status=subscription_data.get("status"),
                next_payment_date=subscription_data.get("next_payment_date"),
                email_token=subscription_data.get("email_token")
            )
            db.add(db_subscription)

            user.subscription_active = True
            user.subscription_code = subscription_data.get("subscription_code")
            if subscription_data.get("email_token"):
                user.email_token = subscription_data.get("email_token")
            user.updated_at = datetime.utcnow()

            db.commit()

            return {
                "status": "success",
                "message": "Subscription created successfully!",
                "data": {
                    "email": email,
                    "subscription_code": subscription_data.get("subscription_code"),
                    "status": subscription_data.get("status"),
                    "next_payment_date": subscription_data.get("next_payment_date")
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/subscription-status/{email}")
async def check_subscription_status(email: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email.lower()).first()

    if not user:
        return {
            "status": "not_found",
            "subscription_active": False
        }

    return {
        "status": "found",
        "subscription_active": user.subscription_active,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "subscription_code": user.subscription_code,
        "email_token": user.email_token,
        "created_at": user.created_at.isoformat() if user.created_at else None
    }

@app.post("/api/cancel-subscription/{email}")
async def cancel_subscription(email: str, db: Session = Depends(get_db)):
    try:
        if not email:
            raise HTTPException(status_code=400, detail="email is required")

        user = db.query(User).filter(User.email == email.lower()).first()
        if not user:
            raise HTTPException(status_code=404, detail="Customer not found")

        if not user.subscription_code:
            raise HTTPException(status_code=400, detail="No active subscription found")

        token = user.email_token
        if not token:
            sub = db.query(Subscription).filter(Subscription.subscription_code == user.subscription_code).first()
            if sub:
                token = sub.email_token

        if not token:
            raise HTTPException(status_code=400, detail="Missing email_token")

        async with httpx.AsyncClient() as client:
            payload = {
                "code": user.subscription_code,
                "token": token
            }

            response = await client.post(
                f"{PAYSTACK_BASE_URL}/subscription/disable",
                json=payload,
                headers=paystack_headers,
                timeout=30.0
            )

            if response.status_code not in (200, 201):
                try:
                    err = response.json()
                except Exception:
                    err = response.text
                raise HTTPException(status_code=400, detail=f"Failed: {err}")

            body = response.json()
            if not body.get("status"):
                raise HTTPException(status_code=400, detail=body.get("message", "Error"))

        user.subscription_active = False
        user.updated_at = datetime.utcnow()
        db.commit()

        return {
            "status": "success",
            "message": "Subscription cancelled successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/webhooks/paystack")
async def paystack_webhook(request: Request, db: Session = Depends(get_db)):
    try:
        raw_body = await request.body()
        payload = json.loads(raw_body)

        signature = request.headers.get("X-Paystack-Signature")
        if not signature:
            return {"status": "error", "message": "Missing signature"}

        computed = hmac.new(
            PAYSTACK_SECRET_KEY.encode(),
            raw_body,
            hashlib.sha512
        ).hexdigest()

        if signature != computed:
            return {"status": "error", "message": "Invalid signature"}

        event = payload.get("event")
        data = payload.get("data") or {}

        if event == "charge.success":
            customer_email = None
            if isinstance(data.get("customer"), dict):
                customer_email = data["customer"].get("email")

            if not customer_email:
                return {"status": "ok"}

            user = db.query(User).filter(User.email == customer_email).first()
            if user:
                authorization = data.get("authorization") or {}
                if authorization.get("authorization_code"):
                    user.authorization_code = authorization.get("authorization_code")
                    user.first_authorization = True
                user.subscription_active = True
                user.last_payment_date = datetime.utcnow()
                user.updated_at = datetime.utcnow()
                db.commit()

            payment_log = PaymentLog(
                email=customer_email,
                reference=data.get("reference"),
                amount=data.get("amount"),
                status="success",
                event_type="charge.success",
                metadata_json=json.dumps(data)
            )
            db.add(payment_log)
            db.commit()

        elif event == "subscription.disable" or event == "subscription.not_renew":
            customer_email = None
            if isinstance(data.get("customer"), dict):
                customer_email = data["customer"].get("email")

            if customer_email:
                user = db.query(User).filter(User.email == customer_email).first()
                if user:
                    user.subscription_active = False
                    user.updated_at = datetime.utcnow()
                    db.commit()

        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        db.rollback()
        return {"status": "error", "message": str(e)}

@app.get("/api/admin/customers")
async def get_all_customers(db: Session = Depends(get_db)):
    users = db.query(User).all()

    customers = [
        {
            "email": user.email,
            "name": f"{user.first_name} {user.last_name}",
            "subscription_active": user.subscription_active,
            "subscription_code": user.subscription_code,
            "created_at": user.created_at.isoformat() if user.created_at else None
        }
        for user in users
    ]

    return {
        "total_customers": len(customers),
        "customers": customers
    }

# CAREER API ENDPOINTS 

@app.get("/api-info")
async def api_info():
    return JSONResponse({
        "api_name": "Motherboard Career Assistant API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "career": [
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
            ],
            "payment": [
                "POST /api/customers",
                "POST /api/initialize-payment",
                "POST /api/verify-payment",
                "POST /api/create-subscription",
                "GET /api/subscription-status/{email}",
                "POST /api/cancel-subscription/{email}",
                "POST /api/webhooks/paystack",
                "GET /api/admin/customers"
            ]
        }
    })

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

# Updated /jobs endpoint with proper location parsing and industry filter

@app.get("/jobs")
async def get_all_available_jobs(session: str = Depends(verify_session)):
    try:
        jobs = get_all_jobs()  # This now loads fresh from Google Sheet each time
        logger.info(f"Fetching all {len(jobs)} available jobs")
        
        # Extract unique locations, industries, and job types from actual data
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
            
            # Extract and parse location - only split by comma
            location = job_dict.get("Location", "N/A")
            if location and location != "N/A":
                # Only split by comma - keeps "Africa (Remote – Africa-based)" together
                location_parts = location.split(',')
                
                for part in location_parts:
                    clean_location = part.strip()
                    if clean_location and clean_location != 'N/A':
                        # Keep the location as-is, just trim whitespace
                        unique_locations.add(clean_location)
            
            # Extract industry
            industry = job_dict.get("Industry", "N/A")
            if industry and industry != "N/A":
                unique_industries.add(industry.strip())
            
            # Extract job type
            job_type = job_dict.get("Job Type", "N/A")
            if job_type and job_type != "N/A":
                unique_job_types.add(job_type.strip())
            
            complete_job = {
                "job_id": idx,
                "job_title": job_dict.get("Job Title", "N/A"),
                "company_name": job_dict.get("Company Name", job_dict.get("Company", "N/A")),
                "job_type": job_type,
                "industry": industry,
                "job_description": description,
                "location": location,  # Keep original format for display
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
        
        # Sort for better UX
        sorted_locations = sorted(list(unique_locations))
        sorted_industries = sorted(list(unique_industries))
        sorted_job_types = sorted(list(unique_job_types))
        
        logger.info(f"Successfully formatted {len(complete_jobs)} jobs")
        logger.info(f"Found {len(sorted_locations)} unique locations: {sorted_locations}")
        logger.info(f"Found {len(sorted_industries)} unique industries: {sorted_industries}")
        logger.info(f"Found {len(sorted_job_types)} unique job types: {sorted_job_types}")
        
        return JSONResponse({
            "jobs": complete_jobs,
            "total_jobs": len(complete_jobs),
            "message": f"Showing all {len(complete_jobs)} available job(s).",
            "available_filters": {
                "countries": sorted_locations,    # Dynamic from actual data
                "industries": sorted_industries,  # Dynamic from actual data
                "job_types": sorted_job_types     # Dynamic from actual data
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
        
        # Get jobs from backend
        if not user_query.strip():
            jobs = get_all_jobs()
        else:
            jobs = find_jobs_from_sentence(user_query)
        
        # Extract unique values for filter options
        unique_locations = set()
        unique_industries = set()
        unique_job_types = set()
        
        # Convert to the same format as /jobs endpoint
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
            
            # Collect unique values for filters
            if location and location != "N/A":
                # Only split by comma
                location_parts = location.split(',')
                for part in location_parts:
                    clean_location = part.strip()
                    if clean_location and clean_location != 'N/A':
                        unique_locations.add(clean_location)
            
            if industry and industry != "N/A":
                unique_industries.add(industry.strip())
            
            if job_type and job_type != "N/A":
                unique_job_types.add(job_type.strip())
            
            complete_job = {
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
            }
            
            complete_jobs.append(complete_job)
        
        # NOW filter using the correct lowercase keys
        # Country filter - match exact location strings (split by comma)
        if country_filter:
            complete_jobs = [
                job for job in complete_jobs
                if job.get("location") and any(
                    country_filter == loc.strip()
                    for loc in job.get("location", "").split(',')
                )
            ]
        
        # Industry filter
        if industry_filter:
            complete_jobs = [
                job for job in complete_jobs
                if industry_filter.lower() in str(job.get("industry", "")).lower()
            ]
        
        # Job type filter
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
        
        # Sort for response
        sorted_locations = sorted(list(unique_locations))
        sorted_industries = sorted(list(unique_industries))
        sorted_job_types = sorted(list(unique_job_types))
        
        return JSONResponse({
            "jobs": complete_jobs,
            "total_found": len(complete_jobs),
            "filters_applied": {
                "query": user_query if user_query else None,
                "country": country_filter if country_filter else None,
                "industry": industry_filter if industry_filter else None,
                "job_type": job_type_filter if job_type_filter else None
            },
            "message": f"Found {len(complete_jobs)} job(s) matching {filter_text}.",
            "available_filters": {
                "countries": sorted_locations,    # Dynamic from actual data
                "industries": sorted_industries,  # Dynamic from actual data
                "job_types": sorted_job_types     # Dynamic from actual data
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
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve assessment statistics"}
        )


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
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve country information"}
        )


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
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to submit feedback"}
        )


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
            content={
                "error": "Failed to reload vector store", 
                "details": str(e)
            }
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
    logger.info("Features: Career Guidance + Payment Management")
    logger.info("Privacy Protection: ENABLED")
    logger.info("Session Management: ENABLED")
    uvicorn.run(app, host="0.0.0.0", port=8000) 