import logging
from typing import Dict, List
from llm_utils import get_ai_response

logger = logging.getLogger(__name__)


class CareerFeedbackGenerator:
    """
    SIMPLE VERSION - Returns only:
    1. Strengths paragraph
    2. Improvements paragraph
    3. Scores breakdown
    
    No fluff, no generic output, straight to the point.
    """
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    def generate_detailed_feedback(
        self, 
        match_result: Dict, 
        job_title: str,
        company: str,
        cv_content: str,
        job_description: str
    ) -> Dict:
        """
        Generate simple, focused feedback with just:
        - Strengths (paragraph)
        - Improvements (paragraph)
        - Scores
        """
        try:
            # Extract missing_keywords from match_result if not provided
            missing_keywords = match_result.get("missing_keywords", [])
            
            # Filter generic keywords
            filtered_keywords = self._filter_generic_keywords(missing_keywords)
            
            # Get strengths paragraph
            strengths_para = self._generate_strengths_paragraph(
                match_result["strengths"],
                job_title
            )
            
            # Get improvements paragraph  
            improvements_para = self._generate_improvements_paragraph(
                filtered_keywords,
                job_title,
                job_description,
                match_result["breakdown"]
            )
            
            return {
                "match_percentage": match_result["overall_match"],
                "match_level": match_result.get("match_level", "Moderate"),
                "strengths": strengths_para,
                "improvements": improvements_para,
                "scores": {
                    "keyword_match": match_result["breakdown"]["keyword_match"],
                    "skills_match": match_result["breakdown"]["skills_match"],
                    "experience_match": match_result["breakdown"]["experience_match"],
                    "semantic_similarity": match_result["breakdown"]["semantic_similarity"]
                }
            }
        
        except Exception as e:
            logger.error(f"Error generating simple feedback: {e}")
            return self._fallback_simple_feedback(match_result)
    
    def _generate_strengths_paragraph(
        self,
        strengths: List[str],
        job_title: str
    ) -> str:
        """Generate a single paragraph about strengths."""
        try:
            if not strengths:
                return "Your CV doesn't clearly highlight any standout strengths for this role."
            
            strengths_str = ", ".join(strengths[:5])
            
            prompt = f"""You are evaluating a candidate's CV for a {job_title} role.

            Their stated strengths are: {strengths_str}

            Write exactly 2-3 short sentences explaining how these specific strengths 
            directly apply to a {job_title} position. 

            Requirements:
            - Reference the actual strengths listed
            - Explain concrete, practical value they bring
            - Avoid: "innovative", "excel in", "high-quality", "best-in-class"
            - For tech roles: explain technical value (e.g., "Python enables building scalable systems")
            - For non-tech roles: explain business/process value (e.g., "Negotiation skills demonstrated through closing deals")
            - For management/mixed roles: focus on transferable, measurable impact
            - Don't start with "This candidate", start with "You..."
            """
            
            response = get_ai_response(
                self.openai_client,
                [{"role": "user", "content": prompt}],
                max_tokens=120
            )
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"Strengths paragraph error: {e}")
            return self._fallback_strengths(strengths)
    
    def _generate_improvements_paragraph(
        self,
        missing_keywords: List[str],
        job_title: str,
        job_description: str,
        breakdown: Dict
    ) -> str:
        """Generate a single paragraph about what to improve."""
        try:
            if not missing_keywords:
                missing_keywords = ["skills not clearly highlighted in your CV"]
            
            keywords_str = ", ".join(missing_keywords[:5])
            
            # Find weakest area
            weakest_area = min(breakdown.items(), key=lambda x: x[1])[0]
            weakest_score = breakdown[weakest_area]
            weakest_area_readable = weakest_area.replace('_', ' ').title()
            
            prompt = f"""You are giving career advice to a candidate applying for a {job_title} role.

            Their weakest area is: {weakest_area_readable} (score: {weakest_score:.0f}%)
            Missing keywords/skills: {keywords_str}
            Job excerpt: {job_description[:400]}

            Write exactly 2-3 short sentences with SPECIFIC, ACTIONABLE advice.

            Focus on:
            1. What specific skill/knowledge to develop (be exact: "Learn Docker", "Build rapport with C-suite clients", "Master project budgeting")
            2. Why it matters for THIS role (reference job posting)
            3. How to demonstrate/develop it (concrete steps: "Build projects", "Lead cross-team initiatives", "Get XYZ certification")
            4. Don't start with "This candidate", start with "You..."
            
            Avoid: vague advice ("learn more", "practice", "improve"), generic phrases
            Be concrete: name actual skills, tools, credentials, or experiences"""
            
            response = get_ai_response(
                self.openai_client,
                [{"role": "user", "content": prompt}],
                max_tokens=120
            )
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"Improvements paragraph error: {e}")
            return self._fallback_improvements(missing_keywords, job_title)
    
    def _filter_generic_keywords(self, keywords: List[str]) -> List[str]:
        """Remove generic words, keep only real skills."""
        generic_words = {
            'you', 'and', 'or', 'the', 'a', 'an', 'is', 'are', 'be', 'been',
            'have', 'has', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'may', 'might', 'must', 'shall', 'was', 'were', 'of', 'to', 'for',
            'in', 'on', 'at', 'by', 'with', 'from', 'up', 'about', 'into', 'through',
            'improving', 'expert', 'administration', 'open', 'high', 'low', 'new',
            'backend\n\nwe'  # Broken keyword from your example
        }
        
        filtered = []
        for kw in keywords:
            kw_lower = kw.lower().strip()
            if kw_lower and kw_lower not in generic_words and len(kw_lower) > 2:
                # Clean up the keyword
                kw_clean = kw_lower.replace('\n', ' ').strip()
                if kw_clean not in [k.lower() for k in filtered]:
                    filtered.append(kw)
        
        return filtered
    
    def _fallback_strengths(self, strengths: List[str]) -> str:
        """Fallback if LLM fails."""
        if strengths:
            top_3 = ", ".join(strengths[:3])
            return f"You have demonstrated skills in {top_3}. These technical foundations will help you handle the core responsibilities of this role."
        return "Your CV has some relevant background for this position."
    
    def _fallback_improvements(self, keywords: List[str], job_title: str) -> str:
        """Fallback if LLM fails."""
        if keywords:
            top_skills = ", ".join(keywords[:3])
            return f"To strengthen your candidacy for {job_title}, prioritize gaining experience with {top_skills}. These are mentioned prominently in the job posting and would significantly improve your fit."
        return f"Review your CV to ensure all relevant skills and experience for {job_title} are clearly highlighted."
    
    def _fallback_simple_feedback(self, match_result: Dict) -> Dict:
        """Complete fallback if everything fails."""
        return {
            "match_percentage": match_result["overall_match"],
            "match_level": "See scores below",
            "strengths": "Review your strengths in the scores breakdown.",
            "improvements": "Review the missing keywords and scores to identify improvement areas.",
            "scores": match_result["breakdown"]
        }