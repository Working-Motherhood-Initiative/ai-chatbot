import re
import logging
from typing import Dict
from datetime import datetime
from fastapi import HTTPException

logger = logging.getLogger(__name__)

def remove_personal_info_from_cv(cv_text: str) -> str:
    try:
        if not cv_text or not cv_text.strip():
            logger.warning("Empty CV text provided for privacy protection")
            return ""
            
        cleaned_text = cv_text
        
        # 1. Remove emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        cleaned_text = re.sub(email_pattern, '[EMAIL REMOVED]', cleaned_text)
        
        # 2. Remove phone numbers
        phone_patterns = [
            r'\+?\d{1,4}[\s\-\(\)]?\(?\d{1,4}\)?[\s\-]?\d{1,4}[\s\-]?\d{1,9}',
            r'\b\d{3}[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b',
            r'\+\d{1,3}\s?\d{1,4}\s?\d{1,4}\s?\d{1,9}',
            r'\(\d{3}\)\s?\d{3}[\s\-]?\d{4}',
        ]
        
        for pattern in phone_patterns:
            cleaned_text = re.sub(pattern, '[PHONE REMOVED]', cleaned_text)
        
        # 3. Remove addresses
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