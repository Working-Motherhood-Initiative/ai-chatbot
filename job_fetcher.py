import os
import pandas as pd
import numpy as np
import openai
import logging
from dotenv import load_dotenv
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global storage for jobs
jobs_data = []

def is_deadline_expired(deadline_str):
    """
    Check if a job posting deadline has expired.
    
    Supports multiple date formats:
    - "2025-12-31"
    - "12/31/2025"
    - "31-12-2025"
    - "Dec 31, 2025"
    
    Returns: True if expired, False if still active
    """
    if not deadline_str or deadline_str.strip() == "":
        # If no deadline specified, assume job is still active
        logger.info("No deadline specified - treating as active posting")
        return False
    
    deadline_str = str(deadline_str).strip()
    today = datetime.now().date()
    
    # Try different date formats
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
    
    # If no format matched, log warning but don't filter out
    logger.warning(f"Could not parse deadline format: {deadline_str}")
    return False


def get_all_jobs(exclude_expired=True):
    """
    Load jobs from storage.
    
    Args:
        exclude_expired: If True, filters out jobs with expired deadlines
    
    Returns: List of jobs (with expired ones removed if exclude_expired=True)
    """
    global jobs_data
    
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


def preload_job_embeddings():
    """Load jobs and precompute embeddings for semantic search."""
    global jobs_data
    
    try:
        # Read from CSV with absolute path
        import os
        csv_path = os.path.join(os.path.dirname(__file__), "jobs.csv")
        logger.info(f"Loading jobs from: {csv_path}")
        df = pd.read_csv(csv_path)
        df.fillna("", inplace=True)
        logger.info(f"Loaded {len(df)} jobs from CSV")

        # Create embeddings for Job Title + Description
        embeddings = embed_text_batch(
            (df["Job Title"] + " " + df["Job Description (Brief Summary)  "]).tolist()
        )

        df["embedding"] = embeddings
        jobs_data = df.to_dict(orient="records")
        
        # Log deadline info
        total_jobs = len(jobs_data)
        expired_count = sum(1 for job in jobs_data if is_deadline_expired(job.get("Application Deadline", "")))
        active_count = total_jobs - expired_count
        
        logger.info(f"Successfully loaded {total_jobs} jobs")
        logger.info(f"  - Active jobs: {active_count}")
        logger.info(f"  - Expired jobs: {expired_count}")
        
    except Exception as e:
        logger.error(f"Error in preload_job_embeddings: {e}")
        raise


def embed_text_batch(text_list):
    """Get embeddings for a batch of texts."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text_list
    )
    return [item.embedding for item in response.data]


def embed_text(text):
    """Get embedding for a single text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def extract_filters(user_query):
    """Use GPT to extract job type, role, and location from a free-text query."""
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
        
        # Try to find JSON-like structure in the response
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
    """Extract relevant keywords from user query using GPT."""
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
        # Split and clean keywords
        keyword_list = [k.strip().lower() for k in keywords.split(',')]
        logger.info(f"Extracted keywords: {keyword_list}")
        return keyword_list

    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

def find_jobs_from_sentence(user_query, exclude_expired=True):
    """
    Enhanced hybrid job search with weighted scoring.
    
    Args:
        user_query: Search query from user
        exclude_expired: If True, excludes jobs with expired deadlines
    
    Returns: Top 5 matching jobs (excluding expired if requested)
    """
    filters = extract_filters(user_query)
    jobs = get_all_jobs(exclude_expired=exclude_expired)  # Filter expired jobs here
    
    if not jobs:
        logger.warning("No active jobs available after filtering")
        return []
    
    query_embedding = embed_text(user_query)

    results = []
    for job in jobs:
        # 1. Base semantic score (40% weight)
        semantic_score = cosine_similarity(query_embedding, job["embedding"]) * 0.4

        # 2. Keyword matching score (30% weight)
        keywords = extract_keywords_from_query(user_query)
        description = job.get("Job Description (Brief Summary)  ", "").lower()
        skills = job.get("Skills Required", "").lower()
        keyword_matches = sum(1 for kw in keywords if kw in description or kw in skills)
        keyword_score = (keyword_matches / max(len(keywords), 1)) * 0.3

        # 3. Filter matching score (30% weight)
        filter_score = 0.0
        if filters["preference"].lower() in job.get("Job Type", "").lower():
            filter_score += 0.15
        if filters["location"].lower() in job.get("Location", "").lower():
            filter_score += 0.15

        # Calculate final weighted score
        final_score = semantic_score + keyword_score + filter_score

        if final_score >= 0.5:  # Adjusted threshold
            results.append((final_score, job))

    results.sort(key=lambda x: x[0], reverse=True)
    return [job for score, job in results[:5]]  # Return top 5 matches