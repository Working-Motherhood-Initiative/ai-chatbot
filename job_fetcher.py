import os
import pandas as pd
import numpy as np
import openai
import logging
from dotenv import load_dotenv
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
GOOGLE_CREDENTIALS_FILE = "gcreds.json"

def setup_google_sheets():
    try:
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_FILE, scopes=scopes)
        client_gs = gspread.authorize(creds)
        logger.info("Successfully connected to Google Sheets")
        return client_gs
    except Exception as e:
        logger.error(f"Error setting up Google Sheets: {e}")
        raise

def load_jobs_from_google_sheet():
    try:
        if not GOOGLE_SHEET_ID:
            raise ValueError("GOOGLE_SHEET_ID environment variable not set")
        
        logger.info(f"Attempting to open sheet with ID: {GOOGLE_SHEET_ID}")
        gs_client = setup_google_sheets()
        sheet = gs_client.open_by_key(GOOGLE_SHEET_ID)
        
        # Get the "Jobs" worksheet
        worksheet = sheet.worksheet("Jobs")
        logger.info(f"Successfully opened worksheet: {worksheet.title}")
        
        # Get all values
        data = worksheet.get_all_values()
        logger.info(f"Retrieved {len(data)} rows from sheet (including header)")
        
        if len(data) < 2:
            raise ValueError("Sheet is empty or only contains headers")
        
        headers = data[0]
        rows = data[1:]
        logger.info(f"Headers found: {headers}")
        
        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=headers)
        df.fillna("", inplace=True)
        
        # Filter out empty rows (rows where Job Title is empty)
        df = df[df["Job Title"].str.strip() != ""]
        
        logger.info(f"Successfully loaded {len(df)} jobs from Google Sheet")
        return df
        
    except Exception as e:
        logger.error(f"Error loading jobs from Google Sheet: {type(e).__name__}: {str(e)}", exc_info=True)
        raise

def is_deadline_expired(deadline_str):
    if not deadline_str or deadline_str.strip() == "":
        logger.info("No deadline specified - treating as active posting")
        return False
    
    deadline_str = str(deadline_str).strip()
    today = datetime.now().date()
    
    date_formats = [
        "%Y-%m-%d",      # 2025-12-31
        "%m/%d/%Y",      # 12/31/2025
        "%d-%m-%Y",      # 31-12-2025
        "%d/%m/%Y",      # 31/12/2025
        "%B %d, %Y",     # December 31, 2025
        "%b %d, %Y",     # Dec 31, 2025
        "%Y/%m/%d",      # 2025/12/31
        "%d.%m.%Y",      # 31.12.2025
    ]
    
    for date_format in date_formats:
        try:
            deadline_date = datetime.strptime(deadline_str, date_format).date()
            is_expired = deadline_date < today
            
            if is_expired:
                logger.info(f"Job deadline {deadline_date} has expired (today: {today})")
            else:
                days_left = (deadline_date - today).days
                logger.info(f"Job deadline {deadline_date} is active ({days_left} days left)")
            
            return is_expired
        except ValueError:
            continue
    
    logger.warning(f"Could not parse deadline format: {deadline_str}")
    return False


def get_all_jobs(exclude_expired=True):
    try:
        df = load_jobs_from_google_sheet()
        jobs_data = df.to_dict(orient="records")
        
        if exclude_expired:
            # Filter out expired jobs
            active_jobs = [
                job for job in jobs_data 
                if not is_deadline_expired(job.get("Application Deadline", ""))
            ]
            logger.info(f"Returning {len(active_jobs)} active jobs (excluded {len(jobs_data) - len(active_jobs)} expired)")
            return active_jobs
        else:
            logger.info(f"Returning all {len(jobs_data)} jobs (including expired)")
            return jobs_data
            
    except Exception as e:
        logger.error(f"Error in get_all_jobs: {type(e).__name__}: {str(e)}", exc_info=True)
        return []


def embed_text_batch(text_list):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text_list
    )
    return [item.embedding for item in response.data]


def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def extract_filters(user_query):
    prompt = f"""
    Extract job search filters from the query.
    Return valid JSON only with keys:
    - preference: Job type (Hybrid, Remote, Onsite) or "" if not found
    - job_title: Role or keywords
    - location: Country or city

    Query: "{user_query}"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a job search filter extractor. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        result = response.choices[0].message.content.strip()
        logger.info(f"Extracted filters raw response: {result}")
        
        import re
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            result = json_match.group()
            
        import json
        return json.loads(result)
    except Exception as e:
        logger.error(f"Error extracting filters: {e}")
        return {"preference": "", "job_title": "", "location": ""}

def extract_keywords_from_query(user_query):
    prompt = f"""
    Extract key job-related terms from this query as a comma-separated list.
    Focus on: skills, job types, industries, and roles.
    Return only the comma-separated list, no other text.
    
    Query: "{user_query}"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract keywords from job search queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        keywords = response.choices[0].message.content.strip()
        keyword_list = [k.strip().lower() for k in keywords.split(',')]
        logger.info(f"Extracted keywords: {keyword_list}")
        return keyword_list

    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

def find_jobs_from_sentence(user_query, exclude_expired=True):
    filters = extract_filters(user_query)
    jobs = get_all_jobs(exclude_expired=exclude_expired)
    
    if not jobs:
        logger.warning("No active jobs available after filtering")
        return []
    
    query_embedding = embed_text(user_query)

    results = []
    for job in jobs:
        semantic_score = cosine_similarity(query_embedding, embed_text(job.get("Job Title", "") + " " + job.get("Job Description (Brief Summary)  ", ""))) * 0.4

        keywords = extract_keywords_from_query(user_query)
        description = job.get("Job Description (Brief Summary)  ", "").lower()
        skills = job.get("Skills Required", "").lower()
        keyword_matches = sum(1 for kw in keywords if kw in description or kw in skills)
        keyword_score = (keyword_matches / max(len(keywords), 1)) * 0.3

        filter_score = 0.0
        if filters["preference"].lower() in job.get("Job Type", "").lower():
            filter_score += 0.15
        if filters["location"].lower() in job.get("Location", "").lower():
            filter_score += 0.15

        final_score = semantic_score + keyword_score + filter_score

        if final_score >= 0.5:
            results.append((final_score, job))

    results.sort(key=lambda x: x[0], reverse=True)
    return [job for score, job in results[:5]]