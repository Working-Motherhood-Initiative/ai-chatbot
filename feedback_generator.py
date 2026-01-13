import logging
from typing import Dict, List
from llm_utils import get_ai_response

logger = logging.getLogger(__name__)


class CareerFeedbackGenerator: 
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    def generate_detailed_feedback(self, match_result: Dict, job_title: str,company: str,cv_content: str,job_description: str) -> Dict:
        
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
                    "skills_match": match_result["breakdown"]["skills_match"],
                    "experience_match": match_result["breakdown"]["experience_match"]
                }
            }
        
        except Exception as e:
            logger.error(f"Error generating simple feedback: {e}")
            return self._fallback_simple_feedback(match_result)

    def _generate_strengths_paragraph(self, strengths: List[str], job_title: str) -> str:
        try:
            if not strengths:
                return "Your CV doesn't clearly highlight any standout strengths for this role."
            
            strengths_str = ", ".join(strengths[:5])
            
            prompt = f"""You are a career advisor reviewing a CV for a {job_title} role.

                The CV successfully highlights these strengths: {strengths_str}

                Write exactly 3-4 short sentences covering BOTH:
                1. CV presentation: How these strengths are effectively showcased (e.g., "Your CV prominently features Python in multiple projects")
                2. Professional value: What impact these bring to the role (e.g., "This enables you to build scalable backend systems")

                Requirements:
                - Start with what's GOOD about the CV presentation
                - Then explain the professional value
                - Be specific to the actual strengths listed
                - For tech roles: mention technical capabilities
                - For non-tech roles: mention business impact
                - Don't start with "This candidate", start with "Your CV..."
                - Avoid: "innovative", "excel in", "high-quality", "best-in-class"

                Example structure:
                "Your CV clearly showcases your Python experience across 3 different projects. This demonstrates your ability to build scalable systems that can handle enterprise-level traffic. Your REST API experience is well-documented, showing you understand modern software architecture."
                """
            
            response = get_ai_response(
                self.openai_client,
                [{"role": "user", "content": prompt}],
                max_tokens=150
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
        try:
            if not missing_keywords:
                missing_keywords = ["skills not clearly highlighted in your CV"]
            
            keywords_str = ", ".join(missing_keywords[:5])
            
            # Find weakest area
            weakest_area = min(breakdown.items(), key=lambda x: x[1])[0]
            weakest_score = breakdown[weakest_area]
            weakest_area_readable = weakest_area.replace('_', ' ').title()
            
            prompt = f"""You are a career advisor helping someone improve their CV for a {job_title} role.

                Their weakest area is: {weakest_area_readable} (score: {weakest_score:.0f}%)
                Missing from CV: {keywords_str}
                Job requirements excerpt: {job_description[:400]}

                Write exactly 4-5 short sentences with TWO types of advice:

                1. IMMEDIATE CV IMPROVEMENTS (what to add/highlight NOW):
                - "Add [specific skill] to your skills section"
                - "Highlight any experience with [tool] in your project descriptions"
                - "Reframe your [role] experience to emphasize [keyword]"
                - "Include metrics showing [skill] impact if you have them"

                2. PROFESSIONAL DEVELOPMENT (if they genuinely lack the skill):
                - "If you haven't used [tool], consider taking a quick online course"
                - "Build a small project using [technology] to gain practical experience"
                - "Seek opportunities at your current job to work with [skill]"

                Requirements:
                - Start with CV improvements (things they can do TODAY)
                - Then mention skill development (things they need to learn)
                - Be SPECIFIC: name exact skills, tools, sections to update
                - Don't start with "This candidate", start with "Your CV..."
                - Distinguish between "you have it but didn't mention it" vs "you need to learn it"

                Example structure:
                "Your CV doesn't mention Docker or Kubernetes, which are critical for this role. If you have container experience, add it to your skills section and describe any deployment work in your project descriptions. If you're new to containerization, start with Docker's official tutorial and deploy a simple application. Also, reframe your backend experience to emphasize scalability and cloud infrastructure, which aligns with the job's focus on distributed systems."
                """
            
            response = get_ai_response(
                self.openai_client,
                [{"role": "user", "content": prompt}],
                max_tokens=180
            )
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"Improvements paragraph error: {e}")
            return self._fallback_improvements(missing_keywords, job_title)
    
    def _filter_generic_keywords(self, keywords: List[str]) -> List[str]:
        generic_words = {
            'you', 'and', 'or', 'the', 'a', 'an', 'is', 'are', 'be', 'been',
            'have', 'has', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'may', 'might', 'must', 'shall', 'was', 'were', 'of', 'to', 'for',
            'in', 'on', 'at', 'by', 'with', 'from', 'up', 'about', 'into', 'through',
            'improving', 'expert', 'administration', 'open', 'high', 'low', 'new',
            'backend\n\nwe'
        }
        
        filtered = []
        for kw in keywords:
            kw_lower = kw.lower().strip()
            if kw_lower and kw_lower not in generic_words and len(kw_lower) > 2:
                kw_clean = kw_lower.replace('\n', ' ').strip()
                if kw_clean not in [k.lower() for k in filtered]:
                    filtered.append(kw)
        
        return filtered
    
    def _fallback_strengths(self, strengths: List[str]) -> str:
        if strengths:
            top_3 = ", ".join(strengths[:3])
            return f"Your CV effectively highlights {top_3}, which are directly relevant to this role. These skills demonstrate you have the technical foundation to handle the core responsibilities. Consider adding specific examples or metrics showing how you've applied these skills in past projects."
        return "Your CV has some relevant background for this position. Consider restructuring it to more prominently feature skills and experience that match the job requirements."
    
    def _fallback_improvements(self, keywords: List[str], job_title: str) -> str:
        if keywords:
            top_skills = ", ".join(keywords[:3])
            return f"Your CV is missing key terms like {top_skills} that appear in the job posting. First, check if you have any experience with these - if so, add them to your skills section and mention them in relevant project descriptions. If you're genuinely missing these skills, prioritize learning them through online courses or hands-on projects, as they're critical for the {job_title} role."
        return f"To improve your CV for this {job_title} position, ensure all relevant skills and experiences are clearly highlighted with specific examples. Review the job description and add any matching keywords you may have overlooked in your current CV."
    
    def _fallback_simple_feedback(self, match_result: Dict) -> Dict:
        return {
            "match_percentage": match_result["overall_match"],
            "match_level": "See scores below",
            "strengths": "Your CV contains relevant skills for this position. Review the scores breakdown to see which areas are strongest.",
            "improvements": "To strengthen your application, review the missing keywords and update your CV to include relevant skills you possess. For skills you're missing, consider targeted professional development.",
            "scores": match_result["breakdown"]
        }