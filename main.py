from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends, status
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
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import logging
import json
from typing import Optional, Dict, List, Tuple
import re
import base64

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
        "https://ai-chatbot2-tjm1.onrender.com",
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

# Google Sheets setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
try:
    google_creds_b64 = os.getenv('GOOGLE_CREDENTIALS')
    if not google_creds_b64:
        raise Exception("GOOGLE_CREDENTIALS not found in environment variables")
    
    creds_dict = json.loads(base64.b64decode(google_creds_b64))
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    gs_client = gspread.authorize(creds)
    logger.info("Successfully initialized Google Sheets client")
except Exception as e:
    logger.error(f"Failed to initialize Google Sheets client: {e}")
    raise

# Enhanced FAQ knowledge base
FAQ_DATA = {
    # CV and Resume Help
    "cv_gaps": "Career gaps are completely normal, especially for mothers. Here's how to handle them: 1) Be honest but brief about the gap 2) Focus on any skills you developed during the break 3) Highlight volunteer work, courses, or freelance projects 4) Use a functional resume format to emphasize skills over timeline 5) Consider a brief explanation like 'Career break for family responsibilities'",
    
    "cv_tips": "Key CV tips for mothers: 1) Use a professional email address 2) Include a strong personal statement highlighting your unique value 3) Quantify achievements where possible 4) Include transferable skills gained through parenting 5) Keep it to 2-3 pages maximum 6) Tailor it for each application 7) Include relevant volunteer work or side projects",
    
    "cv_format": "Best CV formats for career returners: 1) Functional/Skills-based format - emphasizes skills over chronological work history 2) Combination format - highlights both skills and work experience 3) Avoid gaps by using years only (not months) 4) Consider a career summary at the top 5) Use clear, professional formatting with consistent fonts",

    # Career Options and Flexibility
    "flexible_careers": "Great flexible career options for moms include: Remote roles in tech (customer support, content writing, virtual assistance), Freelance work (graphic design, consulting, tutoring), Part-time positions in education or healthcare, E-commerce and online businesses, Project-based consulting in your field of expertise.",
    
    "remote_work": "Remote work opportunities for mothers: 1) Virtual assistance and administrative support 2) Content creation and copywriting 3) Online tutoring and teaching 4) Digital marketing and social media management 5) Graphic design and web development 6) Customer service roles 7) Data entry and bookkeeping 8) Translation services 9) Online counseling or coaching",
    
    "part_time_careers": "Part-time career options: 1) Teaching and education support 2) Healthcare roles (nursing, therapy) 3) Retail and customer service 4) Accounting and bookkeeping 5) Consulting in your expertise area 6) Event planning 7) Real estate 8) Freelance writing or editing 9) Childcare services 10) Administrative support",
    
    "freelance_tips": "Starting freelance work: 1) Identify your marketable skills 2) Create profiles on platforms like Upwork, Fiverr, or LinkedIn 3) Build a portfolio showcasing your work 4) Start with competitive rates to build reviews 5) Network with other professionals 6) Set clear boundaries and work hours 7) Keep detailed records for taxes 8) Always have contracts in place",

    # Maternity Rights - Africa Focus
    "maternity_ghana": "Ghana Maternity Rights: 1) 12 weeks paid maternity leave (Labour Act 2003) 2) Cannot be dismissed during pregnancy or maternity leave 3) Right to return to same position 4) Nursing breaks during working hours 5) Paternity leave: 2 weeks for fathers 6) Must notify employer in writing of pregnancy 7) Social Security covers part of maternity benefits 8) Right to pre and postnatal medical care",
    
    "maternity_nigeria": "Nigeria Maternity Rights: 1) 16 weeks maternity leave (12 weeks paid by employer + 4 weeks unpaid) 2) 2 weeks paternity leave 3) Cannot be dismissed during pregnancy/maternity 4) Right to return to same job 5) Nursing mothers entitled to nursing breaks 6) Must give 3 months notice of maternity leave 7) Some states offer additional benefits 8) National Health Insurance covers some maternity costs",
    
    "maternity_kenya": "Kenya Maternity Rights: 1) 4 months (120 days) maternity leave 2) 2 weeks paternity leave 3) Full pay during maternity leave 4) Cannot be dismissed during pregnancy 5) Right to return to same position 6) Nursing breaks for breastfeeding mothers 7) Must notify employer 2 months before expected delivery 8) Adoption leave also available 9) Part-time work arrangements after maternity leave",
    
    "maternity_africa_general": "General maternity protections across Africa: Most African countries provide 12-16 weeks maternity leave, prohibition of dismissal during pregnancy, and right to return to work. However, enforcement varies. Key countries: South Africa (4 months), Egypt (3 months), Morocco (14 weeks), Tunisia (2 months + 4 months unpaid). Always check your specific country's labor laws and company policies.",

    # Work-Life Balance and Returning to Work
    "returning_confidence": "It's normal to feel anxious about returning to work. Here are some tips: 1) Start with networking events or online communities 2) Take a refresher course in your field 3) Consider contract or part-time work first 4) Update your skills with online courses 5) Practice interviewing with friends 6) Remember that your parenting experience has given you valuable skills like multitasking, patience, and organization.",
    
    "work_life_balance": "Achieving work-life balance as a mother: 1) Set clear boundaries between work and personal time 2) Communicate your needs with your employer 3) Use time management tools and techniques 4) Build a support network (family, friends, childcare) 5) Be realistic about expectations 6) Take care of your physical and mental health 7) Consider flexible work arrangements 8) Don't aim for perfection in everything",
    
    "childcare_options": "Childcare solutions for working mothers: 1) Family daycare centers 2) Nannies or au pairs 3) Family members (grandparents, relatives) 4) Childcare co-ops with other parents 5) After-school programs 6) Employer-provided childcare 7) In-home babysitters 8) Consider backup childcare options 9) Look into government childcare support programs",
    
    "guilt_management": "Managing working mother guilt: 1) Remember that working can benefit your children too 2) Focus on quality time, not quantity 3) Be present when you're with your children 4) Create special traditions and routines 5) Talk to other working mothers for support 6) Practice self-compassion 7) Celebrate your achievements 8) Remember you're setting a positive example",

    # Skills Development and Education
    "skill_development": "Developing new skills while parenting: 1) Online courses (Coursera, Udemy, LinkedIn Learning) 2) YouTube tutorials for practical skills 3) Professional certifications in your field 4) Attend virtual conferences and webinars 5) Join professional associations 6) Network with industry professionals 7) Volunteer for skill-building opportunities 8) Micro-learning during small time windows",
    
    "digital_skills": "Essential digital skills for modern careers: 1) Basic computer proficiency (Microsoft Office, Google Workspace) 2) Social media management 3) Basic graphic design (Canva, Adobe) 4) Video conferencing tools (Zoom, Teams) 5) Project management tools (Trello, Asana) 6) Basic data analysis (Excel, Google Sheets) 7) Online communication and collaboration 8) Cloud storage and file sharing",
    
    "certifications": "Valuable certifications for career advancement: 1) Project Management (PMP, CAPM) 2) Digital Marketing (Google Ads, HubSpot) 3) HR certifications (SHRM, CIPD) 4) IT certifications (CompTIA, Microsoft) 5) Language certifications 6) Industry-specific certifications 7) Leadership and management courses 8) Financial management (CFA, CPA)",

    # Interview and Job Search
    "interview_prep": "Interview preparation for returning mothers: 1) Research the company thoroughly 2) Prepare examples using the STAR method 3) Practice common interview questions 4) Prepare questions to ask the interviewer 5) Plan your outfit and route in advance 6) Bring multiple copies of your resume 7) Be ready to address employment gaps positively 8) Practice with mock interviews 9) Arrive 10-15 minutes early",
    
    "job_search_strategies": "Effective job search strategies: 1) Use multiple job boards (LinkedIn, Indeed, company websites) 2) Network actively (professional associations, alumni networks) 3) Consider recruitment agencies 4) Attend job fairs and networking events 5) Tailor applications for each position 6) Follow up on applications professionally 7) Build your online professional presence 8) Consider informational interviews",
    
    "salary_negotiation": "Salary negotiation tips for mothers: 1) Research market rates for your role and location 2) Consider total compensation, not just salary 3) Prepare your case with specific achievements 4) Practice negotiation conversations 5) Don't accept the first offer immediately 6) Be confident in your value 7) Consider non-monetary benefits 8) Know when to walk away 9) Get agreements in writing",

    # Entrepreneurship and Business
    "starting_business": "Starting a business as a mother: 1) Identify a problem you can solve 2) Start small and test your idea 3) Create a simple business plan 4) Understand legal requirements and registrations 5) Build an emergency fund 6) Network with other entrepreneurs 7) Consider online business models 8) Balance family time with business development 9) Seek mentorship and support groups",
    
    "business_funding": "Funding options for mother entrepreneurs: 1) Personal savings and bootstrapping 2) Small business loans from banks 3) Government grants for women entrepreneurs 4) Angel investors and venture capital 5) Crowdfunding platforms 6) Business incubators and accelerators 7) Friends and family funding 8) Microfinance institutions 9) Women-focused funding organizations",

    # Mental Health and Support
    "stress_management": "Managing work stress as a mother: 1) Practice mindfulness and meditation 2) Exercise regularly, even if brief 3) Get adequate sleep when possible 4) Eat nutritious meals 5) Build a strong support network 6) Set realistic expectations 7) Learn to delegate tasks 8) Take regular breaks 9) Seek professional help when needed",
    
    "support_networks": "Building support networks: 1) Connect with other working mothers 2) Join professional women's groups 3) Participate in community organizations 4) Use online forums and social media groups 5) Find a mentor in your field 6) Build relationships with neighbors 7) Maintain friendships 8) Consider professional counseling or coaching",

    # Legal and Rights Information
    "workplace_discrimination": "Addressing workplace discrimination: 1) Document all incidents with dates and details 2) Report to HR or management 3) Know your company's policies 4) Seek legal advice if necessary 5) Contact labor department or equal opportunity commission 6) Keep records of your performance and contributions 7) Build alliances with colleagues 8) Consider mediation before legal action",
    
    "flexible_work_rights": "Rights to flexible working: Many countries now recognize rights to request flexible working arrangements. This includes: 1) Right to request (not guarantee) flexible hours 2) Employers must consider requests seriously 3) Can appeal if request is denied 4) Protection from discrimination for making request 5) Various arrangements: flextime, compressed hours, remote work, job sharing 6) Check your local employment laws for specific rights",

    # Industry-Specific Advice
    "tech_careers": "Technology careers for mothers: 1) Software development and programming 2) UX/UI design 3) Data analysis and data science 4) Cybersecurity 5) Product management 6) Technical writing 7) Quality assurance testing 8) IT support and administration 9) Many offer remote work options and good work-life balance",
    
    "healthcare_careers": "Healthcare careers with flexibility: 1) Nursing (many part-time and flexible options) 2) Physical therapy 3) Occupational therapy 4) Medical coding and billing 5) Healthcare administration 6) Telemedicine roles 7) Mental health counseling 8) Pharmacy technician 9) Health education and promotion",
    
    "education_careers": "Education career opportunities: 1) Teaching (traditional or online) 2) Tutoring and test preparation 3) Curriculum development 4) Educational technology 5) School administration 6) Training and development 7) Educational consulting 8) Special education support 9) Adult education and literacy"
}

# Question filtering system
CAREER_KEYWORDS = {
    'career', 'job', 'work', 'employment', 'resume', 'cv', 'interview', 'salary', 
    'promotion', 'skills', 'experience', 'qualification', 'training', 'education',
    'professional', 'workplace', 'office', 'remote', 'flexible', 'part-time',
    'full-time', 'freelance', 'contract', 'internship', 'apprenticeship',
    'maternity', 'parental', 'childcare', 'family', 'balance', 'returning',
    'break', 'gap', 'mother', 'mom', 'parent', 'child', 'kids',
    'tech', 'healthcare', 'education', 'marketing', 'finance', 'consulting',
    'management', 'administration', 'customer service', 'sales', 'design',
    'schedule', 'hours', 'benefits', 'leave', 'vacation', 'sick', 'pto',
    'insurance', 'pension', 'retirement'
}

IRRELEVANT_TOPICS = {
    'cook', 'recipe', 'food', 'kitchen', 'ingredients', 'meal', 'dinner',
    'breakfast', 'lunch', 'jollof', 'rice', 'soup', 'stew', 'baking',
    'movie', 'music', 'game', 'tv', 'show', 'entertainment', 'celebrity',
    'sports', 'football', 'basketball', 'dance', 'party',
    'weather', 'travel', 'vacation', 'holiday', 'relationship', 'dating',
    'health', 'medical', 'doctor', 'fitness', 'exercise', 'shopping',
    'fashion', 'beauty', 'makeup'
}

def is_career_related(question: str) -> Tuple[bool, str]:
    """Check if a question is career-related."""
    question_lower = question.lower()
    
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'how', 'what', 'when', 'where', 'why', 'can', 'could', 'should', 'would', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'will', 'shall', 'may', 'might', 'must', 'need', 'want', 'like', 'get', 'got', 'make', 'made', 'take', 'took', 'give', 'gave', 'tell', 'told', 'say', 'said', 'know', 'knew', 'think', 'thought', 'see', 'saw', 'come', 'came', 'go', 'went', 'find', 'found', 'help', 'please', 'thank', 'thanks', 'hello', 'hi', 'hey'}
    
    words = set(re.findall(r'\b\w+\b', question_lower))
    meaningful_words = words - common_words
    
    career_matches = meaningful_words.intersection(CAREER_KEYWORDS)
    irrelevant_matches = meaningful_words.intersection(IRRELEVANT_TOPICS)
    
    if irrelevant_matches and not career_matches:
        return False, f"This appears to be about {', '.join(list(irrelevant_matches)[:3])} which is outside my area of expertise."
    
    if career_matches:
        return True, "Career-related question detected"
    
    work_life_phrases = [
        'work life balance', 'working mom', 'working mother', 'career and family',
        'juggling work', 'returning to work', 'career break', 'professional development'
    ]
    
    if any(phrase in question_lower for phrase in work_life_phrases):
        return True, "Work-life balance question detected"
    
    if len(meaningful_words) < 3:
        return True, "Question needs clarification"
    
    return False, "This doesn't appear to be related to career or work topics."

def get_redirect_message(question: str, reason: str) -> str:
    """Generate a helpful redirect message for off-topic questions."""
    messages = [
        f"I'm Motherboard Career Assistant, specialized in helping mothers with career and work-related questions. {reason}",
        "",
        "I can help you with:",
        "• Job searching and career opportunities",
        "• CV/resume reviews and tips", 
        "• Career guidance and path planning",
        "• Work-life balance for mothers",
        "• Returning to work after career breaks",
        "• Flexible work arrangements",
        "• Interview preparation",
        "• Professional development",
        "• Maternity rights in Africa (Ghana, Nigeria, Kenya)",
        "",
        "What career or work-related question can I help you with today?"
    ]
    return "\n".join(messages)

def search_faq(query: str) -> list:
    """Search FAQ data for relevant entries based on query keywords"""
    query_lower = query.lower()
    relevant_faqs = []
    
    for key, answer in FAQ_DATA.items():
        key_words = key.replace("_", " ")
        if (key_words in query_lower or 
            any(word in query_lower for word in key_words.split()) or
            any(word in answer.lower() for word in query_lower.split() if len(word) > 3)):
            relevant_faqs.append({
                "topic": key,
                "answer": answer,
                "relevance_score": len([word for word in query_lower.split() if word in answer.lower()])
            })
    
    relevant_faqs.sort(key=lambda x: x["relevance_score"], reverse=True)
    return relevant_faqs[:3]

def get_faq_response(question: str) -> tuple:
    """Get FAQ response with better matching logic"""
    question_lower = question.lower()
    
    for key, answer in FAQ_DATA.items():
        key_phrases = key.replace("_", " ").split()
        if any(phrase in question_lower for phrase in key_phrases):
            related_topics = [k for k in FAQ_DATA.keys() if k != key][:5]
            return answer, related_topics
    
    relevant_faqs = search_faq(question)
    if relevant_faqs and relevant_faqs[0]["relevance_score"] > 0:
        best_match = relevant_faqs[0]
        related_topics = [faq["topic"] for faq in relevant_faqs[1:]]
        return best_match["answer"], related_topics
    
    return None, list(FAQ_DATA.keys())[:5]

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
            code { background: #e9ecef; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>👩‍💼 Motherboard Career Assistant API</h1>
            <p>Welcome to the Motherboard Career Assistant API - helping mothers navigate work and career!</p>
            
            <h2>🚀 API Status</h2>
            <p><strong>Status:</strong> <span style="color: green;">✅ Running</span></p>
            <p><strong>Version:</strong> 1.0.0</p>
            
            <h2>📋 Available Endpoints</h2>
            
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
                <span class="method">POST</span> <code>/career-path</code>
                <p>Get personalized career guidance</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/subscribe</code>
                <p>Subscribe for job alerts and notifications</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/faq</code>
                <p>Ask questions about career, work-life balance, maternity rights</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/health</code>
                <p>Health check endpoint</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/docs</code>
                <p>Interactive API documentation (Swagger UI)</p>
            </div>
            
            <h2>🔐 Authentication</h2>
            <p>Most endpoints require authentication. Include your API token in the Authorization header:</p>
            <code>Authorization: Bearer YOUR_API_TOKEN</code>
            
            <h2>🌍 Africa Focus</h2>
            <p>Specialized support for working mothers in Ghana, Nigeria, and Kenya with comprehensive maternity rights information.</p>
            
            <h2>📖 Documentation</h2>
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
            f"Hi {user_name}! 👋 Welcome to Motherboard - your career companion for navigating work and motherhood.\n\n"
            "I'm here to help you with:\n"
            "🔍 **Job Search** - Find flexible, remote, and part-time opportunities\n"
            "📄 **CV Review** - Get personalized feedback on your resume\n"
            "🎯 **Career Guidance** - Explore new career paths and opportunities\n"
            "📧 **Job Alerts** - Subscribe to receive relevant job notifications\n"
            "❓ **Support** - Get answers about returning to work, CV gaps, maternity rights\n"
            "🌍 **Africa Focus** - Specialized support for Ghana, Nigeria, and Kenya\n\n"
            "What would you like to start with today?"
        )

    return JSONResponse({
        "response": welcome_message,
        "suggestions": [
            "Upload my CV for review",
            "Search for jobs",
            "What are maternity rights in my country?",
            "Tell me about flexible work options"
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
                "Search for relevant jobs",
                "Get career path guidance", 
                "Subscribe for job alerts"
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
        
        # Remove embedding field from matches
        clean_matches = []
        for job in matches:
            job_dict = dict(job)
            if 'embedding' in job_dict:
                del job_dict['embedding']
            clean_matches.append(job_dict)
            
        logger.info(f"Found {len(clean_matches)} matches")
        
        return JSONResponse({
            "matches": clean_matches,
            "total_found": len(clean_matches),
            "search_query": user_query,
            "suggestions": [
                "Subscribe for job alerts",
                "Get career guidance",
                "Upload CV for better matches"
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

@app.post("/career-path")
async def suggest_career_path(request: Request, token: str = Depends(verify_token)):
    """Provide career guidance from a single free-form query with topic filtering"""
    try:
        data = await request.json()
        query = data.get("query", "").strip()

        if not query:
            return JSONResponse({
                "response": "Please tell me about your background, experience, interests, and work preferences. For example:\n"
                            "'I'm a former marketing manager with 10 years' experience, looking for remote work in sustainability.'",
                "suggestions": [
                    "Tell me about your work experience",
                    "What career change are you considering?",
                    "What type of work environment do you prefer?"
                ]
            })

        # Check if query is career-related
        is_relevant, reason = is_career_related(query)
        
        if not is_relevant:
            return JSONResponse({
                "response": get_redirect_message(query, reason),
                "type": "redirect",
                "suggestions": [
                    "Tell me about your work experience",
                    "What career change are you considering?",
                    "What type of work environment do you prefer?"
                ]
            })

        # Generate career advice
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a supportive career coach specializing in helping African mothers (especially in Ghana, Nigeria, and Kenya) find fulfilling work. "
                    "From the provided query, suggest 2–3 specific, actionable career paths. "
                    "Include practical next steps and be encouraging. Consider the African job market, cultural context, and opportunities for mothers. "
                    "ONLY provide career and professional guidance - do not answer questions about cooking, entertainment, or other non-career topics."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ]
        career_advice = get_ai_response(messages, max_tokens=600)

        # Use the full query to find relevant jobs
        relevant_jobs = find_jobs_from_sentence(query)
        clean_jobs = []
        for job in relevant_jobs[:3]:
            job_dict = dict(job)
            job_dict.pop("embedding", None)
            clean_jobs.append(job_dict)

        return JSONResponse({
            "response": career_advice,
            "relevant_jobs": clean_jobs,
            "next_steps": [
                "Explore recommended jobs",
                "Subscribe for job alerts",
                "Upload your CV for personalized feedback"
            ]
        })

    except Exception as e:
        logger.error(f"Career path error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Sorry, I couldn't provide career guidance right now."}
        )

@app.post("/subscribe")
async def subscribe_user(request: Request, token: str = Depends(verify_token)):
    """Subscribe user for job alerts"""
    try:
        data = await request.json()
        email = data.get("email", "").strip()
        interests = data.get("interests", "Not specified")
        job_types = data.get("job_types", [])
        name = data.get("name", "")
        location = data.get("location", "")

        if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return JSONResponse(
                status_code=400,
                content={"error": "Please provide a valid email address."}
            )

        try:
            spreadsheet_id = os.getenv("SPREADSHEET_ID")
            spreadsheet = gs_client.open_by_key(spreadsheet_id)
            sheet = spreadsheet.worksheet("Subscribers")
            
            # Add subscriber data
            sheet.append_row([
                datetime.now().isoformat(),
                name,
                email,
                interests,
                ", ".join(job_types) if job_types else "All types",
                location
            ])

            return JSONResponse({
                "message": f"Successfully subscribed! You'll receive job alerts matching your interests at {email}.",
                "response": "Thank you for subscribing! 🎉 You'll be the first to know about new opportunities that match your preferences.",
                "next_steps": [
                    "Upload your CV for better matches",
                    "Explore current job listings",
                    "Get career guidance"
                ]
            })

        except Exception as e:
            logger.error(f"Subscription error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Sorry, there was an issue with your subscription. Please try again."}
            )

    except Exception as e:
        logger.error(f"Subscribe endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Sorry, subscription failed. Please try again."}
        )

@app.post("/faq")
async def handle_faq(request: Request, token: str = Depends(verify_token)):
    """Handle FAQ and general support questions with enhanced topic filtering and search"""
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        
        if not question:
            return JSONResponse({
                "response": "Please ask me a question about careers, work, or professional development!",
                "suggestions": [
                    "What are maternity rights in Ghana?",
                    "How do I handle career gaps in my CV?", 
                    "What flexible career options are available?",
                    "How can I start a business as a mother?"
                ]
            })

        # Check if question is career-related
        is_relevant, reason = is_career_related(question)
        
        if not is_relevant:
            return JSONResponse({
                "response": get_redirect_message(question, reason),
                "type": "redirect",
                "suggestions": [
                    "Maternity rights in Africa",
                    "Flexible work arrangements", 
                    "Career guidance for mothers",
                    "Starting a business as a mom"
                ]
            })

        # Use enhanced FAQ matching
        faq_response, related_topics = get_faq_response(question)

        if faq_response:
            return JSONResponse({
                "response": faq_response,
                "type": "faq",
                "related_topics": related_topics,
                "suggestions": [
                    f"Tell me about {related_topics[0].replace('_', ' ')}" if related_topics else "Search for jobs",
                    "Upload CV for feedback",
                    "Get career guidance"
                ]
            })

        # Use AI for other career-related questions with enhanced context
        enhanced_context = """
        You are Motherboard Career Assistant, specialized in helping African mothers (especially in Ghana, Nigeria, and Kenya) with career and work-related challenges. 
        
        You have access to comprehensive information about:
        - Maternity rights and policies across Africa
        - Flexible career options and remote work
        - CV writing and interview preparation
        - Work-life balance strategies
        - Entrepreneurship and business development
        - Skills development and certifications
        - Industry-specific career advice
        
        IMPORTANT: Only answer questions related to careers, work, professional development, job searching, work-life balance for mothers, maternity rights, and similar professional topics. 
        
        Provide helpful, encouraging advice that's culturally relevant and specific to working mothers in Africa. Include practical next steps and be empowering.
        """

        messages = [
            {
                "role": "system", 
                "content": enhanced_context
            },
            {
                "role": "user", 
                "content": f"Question: {question}"
            }
        ]

        ai_response = get_ai_response(messages, max_tokens=500)

        return JSONResponse({
            "response": ai_response,
            "type": "ai_generated",
            "suggestions": [
                "Learn about maternity rights",
                "Explore flexible careers",
                "Get CV feedback",
                "Subscribe for job alerts"
            ]
        })

    except Exception as e:
        logger.error(f"FAQ error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Sorry, I couldn't answer your question right now. Please try again."}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "service": "Motherboard Career Assistant API"
    }

@app.get("/test-auth")
async def test_auth(token: str = Depends(verify_token)):
    """Test authentication endpoint"""
    return {
        "message": "Authentication successful",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api-info")
async def api_info():
    """Get API information and available endpoints"""
    return {
        "name": "Motherboard Career Assistant API",
        "version": "1.0.0",
        "description": "API for helping mothers navigate work and career",
        "focus": "African mothers in Ghana, Nigeria, and Kenya",
        "features": [
            "Job search and matching",
            "CV analysis and tips",
            "Career path guidance", 
            "FAQ and support",
            "Job alert subscriptions",
            "Maternity rights information"
        ],
        "endpoints": {
            "POST /welcome": "Welcome and onboard users",
            "POST /cv-tips": "CV analysis and improvement tips",
            "POST /search-jobs": "Search job opportunities",
            "POST /career-path": "Get career guidance",
            "POST /subscribe": "Subscribe for job alerts",
            "POST /faq": "Ask career-related questions",
            "GET /health": "Health check",
            "GET /api-info": "API information"
        },
        "timestamp": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
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
    """General exception handler"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)