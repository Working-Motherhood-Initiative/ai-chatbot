import re
import logging
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

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


def calculate_keyword_match(cv_text: str, job_description: str) -> float:
    job_keywords = extract_important_keywords(job_description)
    
    if not job_keywords:
        return 50.0
    
    cv_lower = cv_text.lower()
    matches = sum(1 for keyword in job_keywords if keyword.lower() in cv_lower)
    
    return (matches / len(job_keywords)) * 100


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


def estimate_cv_experience_years(cv_text: str) -> int:
    date_ranges = re.findall(r'(20\d{2}|19\d{2})\s*[-–]\s*(20\d{2}|present|current)', cv_text, re.IGNORECASE)
    
    total_years = 0
    for start_str, end_str in date_ranges:
        start_year = int(start_str)
        end_year = 2025 if end_str.lower() in ['present', 'current'] else int(end_str)
        total_years += max(0, end_year - start_year)
    
    return total_years


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


def calculate_cv_job_match_hybrid(cv_text: str, job_description: str, job_title: str,skills_required: str = "") -> Dict:
    combined_job_text = f"{job_description}\n\nRequired Skills: {skills_required}".strip()
    
    keyword_score = calculate_keyword_match(cv_text, combined_job_text)
    skills_score = calculate_skills_match(cv_text, combined_job_text)
    experience_score = calculate_experience_match(cv_text, combined_job_text, job_title)
    
    
    total_score = (
        keyword_score * 0.10 +      
        skills_score * 0.50 +       
        experience_score * 0.40    
    )

    missing_keywords = find_missing_keywords(cv_text, combined_job_text)
    strengths = identify_strengths(cv_text, combined_job_text)
    
    return {
        "overall_match": round(total_score),
        "breakdown": {
            "keyword_match": round(keyword_score),
            "skills_match": round(skills_score), 
            "experience_match": round(experience_score)
            # semantic_similarity removed
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


def extract_skills_from_cv(cv_text: str, openai_client) -> List[str]:
    try:
        messages = [
            {
                "role": "system", 
                "content": "You are a skill extraction expert. Extract key skills, technologies, and competencies from the privacy-protected CV text. Return only a comma-separated list of skills, no explanations."
            },
            {
                "role": "user", 
                "content": f"Extract skills from this privacy-protected CV:\n{cv_text[:2000]}"
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