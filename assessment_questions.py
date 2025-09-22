from typing import Dict, Any
from datetime import datetime

# Assessment questions data
ASSESSMENT_QUESTIONS = {
    "career_readiness": [
        {
            "id": "public_speaking",
            "question": "How comfortable are you with public speaking or presenting?",
            "options": {
                "very_comfortable": "Very comfortable - I enjoy presenting to groups",
                "somewhat_comfortable": "Somewhat comfortable - I can manage when needed",
                "neutral": "Neutral - It depends on the situation",
                "somewhat_uncomfortable": "Somewhat uncomfortable - I find it challenging",
                "very_uncomfortable": "Very uncomfortable - I avoid it when possible"
            }
        },
        {
            "id": "salary_negotiation",
            "question": "Have you ever negotiated a job salary or raise?",
            "options": {
                "multiple_times": "Yes, multiple times successfully",
                "once_successfully": "Yes, once and it went well",
                "once_unsuccessfully": "Yes, once but it didn't go as hoped",
                "no_but_would": "No, but I would be willing to try",
                "no_and_anxious": "No, and I find the idea intimidating"
            }
        },
        {
            "id": "networking",
            "question": "How do you feel about professional networking?",
            "options": {
                "love_it": "I love meeting new people and building connections",
                "comfortable": "I'm comfortable and see the value in it",
                "neutral": "It's okay - I do it when necessary",
                "find_difficult": "I find it difficult but understand it's important",
                "avoid_it": "I tend to avoid networking situations"
            }
        },
        {
            "id": "skill_confidence",
            "question": "How confident are you in your current professional skills?",
            "options": {
                "very_confident": "Very confident - I know my strengths well",
                "mostly_confident": "Mostly confident with some areas to improve",
                "neutral": "Average confidence level",
                "somewhat_doubtful": "I often doubt my abilities",
                "very_doubtful": "I feel my skills are quite outdated"
            }
        },
        {
            "id": "leadership_experience",
            "question": "Do you have experience leading teams or projects?",
            "options": {
                "extensive": "Extensive experience leading multiple teams/projects",
                "some_formal": "Some formal leadership roles",
                "informal_only": "Only informal leadership (organizing events, groups)",
                "minimal": "Very little leadership experience",
                "none": "No leadership experience"
            }
        },
        {
            "id": "career_goals",
            "question": "How clear are you about your career goals?",
            "options": {
                "very_clear": "Very clear - I have a detailed plan",
                "mostly_clear": "Mostly clear with some flexibility",
                "somewhat_clear": "I have general ideas but need more clarity",
                "unclear": "Quite unclear about my direction",
                "completely_unsure": "Completely unsure what I want to do"
            }
        },
        {
            "id": "interview_confidence",
            "question": "How do you feel about job interviews?",
            "options": {
                "confident": "Confident - I interview well",
                "prepared": "Good when well-prepared",
                "nervous_but_okay": "Nervous but I manage okay",
                "very_nervous": "Very nervous and struggle to perform well",
                "panic": "I panic and find them very difficult"
            }
        },
        {
            "id": "technology_comfort",
            "question": "How comfortable are you with learning new technology or software?",
            "options": {
                "very_comfortable": "Very comfortable - I pick up new tools quickly",
                "comfortable": "Comfortable with some guidance",
                "neutral": "It depends on the complexity",
                "challenging": "I find it challenging but manageable",
                "very_difficult": "Very difficult - I prefer familiar tools"
            }
        },
        {
            "id": "feedback_handling",
            "question": "How do you handle constructive criticism or feedback?",
            "options": {
                "welcome_it": "I actively seek and welcome feedback",
                "handle_well": "I handle it well and use it to improve",
                "neutral": "It depends on how it's delivered",
                "find_difficult": "I find it difficult but try to learn",
                "very_sensitive": "I'm very sensitive to criticism"
            }
        },
        {
            "id": "career_break_confidence",
            "question": "How confident do you feel about re-entering the workforce after a career break?",
            "options": {
                "very_confident": "Very confident - I'm ready to jump back in",
                "mostly_confident": "Mostly confident with some preparation",
                "somewhat_worried": "Somewhat worried about the transition",
                "quite_anxious": "Quite anxious about being competitive",
                "very_anxious": "Very anxious about my prospects"
            }
        }
    ],
    "work_life_balance": [
        {
            "id": "flexible_hours",
            "question": "Do you have access to flexible work hours in your current situation?",
            "options": {
                "full_flexibility": "Yes, I have complete control over my schedule",
                "some_flexibility": "Yes, I have some flexibility within limits",
                "limited_flexibility": "Limited flexibility for specific needs",
                "no_flexibility": "No, my schedule is quite rigid",
                "not_working": "I'm not currently working"
            }
        },
        {
            "id": "family_stress",
            "question": "How often do you feel overwhelmed managing family responsibilities?",
            "options": {
                "rarely": "Rarely - I have good systems in place",
                "occasionally": "Occasionally during busy periods",
                "sometimes": "Sometimes - it's manageable most days",
                "frequently": "Frequently - I often feel stretched thin",
                "constantly": "Almost constantly - I'm always overwhelmed"
            }
        },
        {
            "id": "support_system",
            "question": "How would you describe your family support system for your career?",
            "options": {
                "very_supportive": "Very supportive - they actively encourage my career",
                "mostly_supportive": "Mostly supportive with occasional concerns",
                "neutral": "Neutral - they don't interfere either way",
                "somewhat_unsupportive": "Somewhat unsupportive or skeptical",
                "very_unsupportive": "Very unsupportive or discouraging"
            }
        },
        {
            "id": "childcare_arrangements",
            "question": "How secure do you feel about your childcare arrangements?",
            "options": {
                "very_secure": "Very secure - reliable long-term arrangements",
                "mostly_secure": "Mostly secure with backup plans",
                "somewhat_secure": "Somewhat secure but could be better",
                "insecure": "Not very secure - often scrambling",
                "no_arrangements": "No reliable arrangements in place"
            }
        },
        {
            "id": "personal_time",
            "question": "How often do you get time for yourself each week?",
            "options": {
                "plenty": "Plenty - I prioritize personal time well",
                "adequate": "Adequate amount most weeks",
                "some": "Some time but not as much as I'd like",
                "very_little": "Very little - mostly focused on family/work",
                "none": "Almost no personal time"
            }
        },
        {
            "id": "work_interruptions",
            "question": "How often do family responsibilities interrupt your work time?",
            "options": {
                "rarely": "Rarely - I have clear boundaries",
                "occasionally": "Occasionally for important matters",
                "regularly": "Regularly but manageable",
                "frequently": "Frequently - it's a constant challenge",
                "constantly": "Almost constantly interrupted"
            }
        },
        {
            "id": "partner_support",
            "question": "How much does your partner (if applicable) share household and childcare duties?",
            "options": {
                "equal_partnership": "We share duties equally as partners",
                "mostly_shared": "They help significantly with most tasks",
                "some_help": "They help but I handle most responsibilities",
                "minimal_help": "They provide minimal help",
                "single_parent": "I'm a single parent managing alone"
            }
        },
        {
            "id": "guilt_feelings",
            "question": "How often do you feel guilty about working vs. spending time with family?",
            "options": {
                "never": "Never - I'm comfortable with my choices",
                "rarely": "Rarely - mostly at peace with balance",
                "sometimes": "Sometimes during particularly busy periods",
                "frequently": "Frequently - it's a constant struggle",
                "always": "Almost always - the guilt is overwhelming"
            }
        },
        {
            "id": "energy_levels",
            "question": "How are your energy levels for work after handling family responsibilities?",
            "options": {
                "high_energy": "High - I manage my energy well",
                "good_energy": "Generally good with occasional tired days",
                "moderate_energy": "Moderate - depends on the day",
                "low_energy": "Often low energy for work tasks",
                "exhausted": "Frequently exhausted before work even begins"
            }
        },
        {
            "id": "future_planning",
            "question": "How confident are you about managing career growth with family life?",
            "options": {
                "very_confident": "Very confident - I have a clear plan",
                "mostly_confident": "Mostly confident with some concerns",
                "uncertain": "Uncertain about how to balance both",
                "worried": "Worried one will have to suffer for the other",
                "overwhelmed": "Overwhelmed by the complexity"
            }
        }
    ]
}

# Scoring weights for each response (1-5 scale)
SCORING_WEIGHTS = {
    "career_readiness": {
        "public_speaking": {
            "very_comfortable": 5, 
            "somewhat_comfortable": 4, 
            "neutral": 3, 
            "somewhat_uncomfortable": 2, 
            "very_uncomfortable": 1
        },
        "salary_negotiation": {
            "multiple_times": 5, 
            "once_successfully": 4, 
            "once_unsuccessfully": 3, 
            "no_but_would": 2, 
            "no_and_anxious": 1
        },
        "networking": {
            "love_it": 5, 
            "comfortable": 4, 
            "neutral": 3, 
            "find_difficult": 2, 
            "avoid_it": 1
        },
        "skill_confidence": {
            "very_confident": 5, 
            "mostly_confident": 4, 
            "neutral": 3, 
            "somewhat_doubtful": 2, 
            "very_doubtful": 1
        },
        "leadership_experience": {
            "extensive": 5, 
            "some_formal": 4, 
            "informal_only": 3, 
            "minimal": 2, 
            "none": 1
        },
        "career_goals": {
            "very_clear": 5, 
            "mostly_clear": 4, 
            "somewhat_clear": 3, 
            "unclear": 2, 
            "completely_unsure": 1
        },
        "interview_confidence": {
            "confident": 5, 
            "prepared": 4, 
            "nervous_but_okay": 3, 
            "very_nervous": 2, 
            "panic": 1
        },
        "technology_comfort": {
            "very_comfortable": 5, 
            "comfortable": 4, 
            "neutral": 3, 
            "challenging": 2, 
            "very_difficult": 1
        },
        "feedback_handling": {
            "welcome_it": 5, 
            "handle_well": 4, 
            "neutral": 3, 
            "find_difficult": 2, 
            "very_sensitive": 1
        },
        "career_break_confidence": {
            "very_confident": 5, 
            "mostly_confident": 4, 
            "somewhat_worried": 3, 
            "quite_anxious": 2, 
            "very_anxious": 1
        }
    },
    "work_life_balance": {
        "flexible_hours": {
            "full_flexibility": 5, 
            "some_flexibility": 4, 
            "limited_flexibility": 3, 
            "no_flexibility": 2, 
            "not_working": 3  # Neutral for not working
        },
        "family_stress": {
            "rarely": 5, 
            "occasionally": 4, 
            "sometimes": 3, 
            "frequently": 2, 
            "constantly": 1
        },
        "support_system": {
            "very_supportive": 5, 
            "mostly_supportive": 4, 
            "neutral": 3, 
            "somewhat_unsupportive": 2, 
            "very_unsupportive": 1
        },
        "childcare_arrangements": {
            "very_secure": 5, 
            "mostly_secure": 4, 
            "somewhat_secure": 3, 
            "insecure": 2, 
            "no_arrangements": 1
        },
        "personal_time": {
            "plenty": 5, 
            "adequate": 4, 
            "some": 3, 
            "very_little": 2, 
            "none": 1
        },
        "work_interruptions": {
            "rarely": 5, 
            "occasionally": 4, 
            "regularly": 3, 
            "frequently": 2, 
            "constantly": 1
        },
        "partner_support": {
            "equal_partnership": 5, 
            "mostly_shared": 4, 
            "some_help": 3, 
            "minimal_help": 2, 
            "single_parent": 3  # Neutral for single parents
        },
        "guilt_feelings": {
            "never": 5, 
            "rarely": 4, 
            "sometimes": 3, 
            "frequently": 2, 
            "always": 1
        },
        "energy_levels": {
            "high_energy": 5, 
            "good_energy": 4, 
            "moderate_energy": 3, 
            "low_energy": 2, 
            "exhausted": 1
        },
        "future_planning": {
            "very_confident": 5, 
            "mostly_confident": 4, 
            "uncertain": 3, 
            "worried": 2, 
            "overwhelmed": 1
        }
    }
}

def calculate_assessment_scores(responses: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
   #Calculate career readiness score
    career_total = 0
    career_count = 0
    for q_id, response in responses.get("career_readiness", {}).items():
        if q_id in SCORING_WEIGHTS["career_readiness"]:
            score = SCORING_WEIGHTS["career_readiness"][q_id].get(response, 0)
            career_total += score
            career_count += 1
    
    # Calculate work-life balance score
    balance_total = 0
    balance_count = 0
    for q_id, response in responses.get("work_life_balance", {}).items():
        if q_id in SCORING_WEIGHTS["work_life_balance"]:
            score = SCORING_WEIGHTS["work_life_balance"][q_id].get(response, 0)
            balance_total += score
            balance_count += 1
    
    # Convert to percentage scores (1-5 scale to 0-100)
    career_score = round((career_total / max(career_count, 1)) * 20) if career_count > 0 else 0
    balance_score = round((balance_total / max(balance_count, 1)) * 20) if balance_count > 0 else 0
    overall_score = round((career_score + balance_score) / 2)
    
    return {
        "career_readiness_score": career_score,
        "work_life_balance_score": balance_score,
        "overall_score": overall_score,
        "career_responses": responses.get("career_readiness", {}),
        "balance_responses": responses.get("work_life_balance", {}),
        "raw_scores": {
            "career_total": career_total,
            "career_count": career_count,
            "balance_total": balance_total,
            "balance_count": balance_count
        }
    }

def generate_assessment_feedback(scores: Dict[str, Any]) -> str:
    career_score = scores["career_readiness_score"]
    balance_score = scores["work_life_balance_score"]
    overall_score = scores["overall_score"]
    
    feedback = f"MomFit Assessment Results\n\n"
    feedback += f"Overall Score: {overall_score}/100\n"
    feedback += f"- Career Readiness: {career_score}/100\n"
    feedback += f"- Work-Life Balance: {balance_score}/100\n\n"
    
    # Overall assessment with status
    if overall_score >= 80:
        feedback += "Status: Ready to Thrive!\n"
        feedback += "You're in a strong position with good career confidence and work-life balance management. You're well-positioned to pursue ambitious career goals while maintaining family harmony.\n\n"
    elif overall_score >= 60:
        feedback += "Status: Building Momentum\n"
        feedback += "You have a solid foundation with some areas for focused improvement. With targeted effort, you can reach the next level in both career and life balance.\n\n"
    elif overall_score >= 40:
        feedback += "Status: Growing Strong\n"
        feedback += "You're on a positive path with several opportunities for development. This is an exciting time to focus on building your strengths and addressing key areas.\n\n"
    else:
        feedback += "Status: Foundation Building\n"
        feedback += "This is your starting point - every expert was once a beginner! Focus on small, consistent steps to build confidence and systems that work for your family.\n\n"
    
    # Career readiness insights
    feedback += "Career Readiness Insights:\n"
    if career_score >= 75:
        feedback += "- You demonstrate strong professional confidence and skills\n"
        feedback += "- Consider leveraging your strengths to pursue stretch opportunities or leadership roles\n"
        feedback += "- You could mentor other mothers transitioning back to work\n"
        feedback += "- Look for roles that challenge you and utilize your full potential\n"
    elif career_score >= 50:
        feedback += "- You have good foundational skills with room to build confidence\n"
        feedback += "- Focus on areas where you feel less certain - practice makes progress\n"
        feedback += "- Consider joining professional networks or skill-building workshops\n"
        feedback += "- Update your skills in 1-2 key areas to boost your marketability\n"
    else:
        feedback += "- This is a perfect time to focus on building professional confidence\n"
        feedback += "- Start with small steps like practicing interview skills or updating your CV\n"
        feedback += "- Consider online courses or local workshops to refresh your skills\n"
        feedback += "- Connect with other professional mothers for support and advice\n"
    
    # Work-life balance insights
    feedback += "\nWork-Life Balance Insights:\n"
    if balance_score >= 75:
        feedback += "- You've created a sustainable balance that works for your family\n"
        feedback += "- Your support systems and boundaries are serving you well\n"
        feedback += "- You might be able to take on additional career challenges\n"
        feedback += "- Share your successful strategies with other mothers\n"
    elif balance_score >= 50:
        feedback += "- You're managing well but could benefit from stronger support systems\n"
        feedback += "- Look for ways to reduce overwhelm and create more personal time\n"
        feedback += "- Consider discussing family responsibilities and support needs openly\n"
        feedback += "- Explore flexible work options that could improve your situation\n"
    else:
        feedback += "- You're juggling a lot - be kind to yourself during this phase\n"
        feedback += "- Focus on building one support system at a time\n"
        feedback += "- Remember: asking for help is a sign of wisdom, not weakness\n"
        feedback += "- Prioritize your well-being - you can't pour from an empty cup\n"
    
    # Personalized action steps based on scores
    feedback += "\nRecommended Next Steps:\n"
    
    # Career-focused recommendations
    if career_score < 50:
        feedback += "1. Skill Building: Take one online course in an area that interests you or aligns with your goals\n"
        feedback += "2. Network: Join one professional group, online community, or attend a local meetup\n"
    elif career_score < 75:
        feedback += "1. Opportunity Search: Start actively looking for roles that match your goals and skills\n"
        feedback += "2. Skill Showcase: Update your CV and LinkedIn to highlight your strengths and recent learning\n"
    else:
        feedback += "1. Leadership Growth: Look for opportunities to lead projects or mentor others\n"
        feedback += "2. Strategic Planning: Set ambitious but achievable career goals for the next 12 months\n"
    
    # Balance-focused recommendations
    if balance_score < 50:
        feedback += "3. Support System: Identify one area where you need more help and take steps to get it\n"
        feedback += "4. Self-Care: Schedule 30 minutes of personal time twice this week - non-negotiable\n"
    elif balance_score < 75:
        feedback += "3. Boundary Setting: Establish clearer boundaries between work and family time\n"
        feedback += "4. Efficiency: Streamline one household or work process to save time and energy\n"
    else:
        feedback += "3. Growth Planning: Set one professional goal for the next 3 months\n"
        feedback += "4. Help Others: Share your balance strategies with other mothers in your network\n"
    
    # Encouraging conclusion
    feedback += "\nRemember: Your journey is unique, and every small step forward matters. You're doing amazing work balancing motherhood and career aspirations. Trust yourself, be patient with the process, and celebrate your progress along the way!"
    
    return feedback

def get_assessment_instructions() -> Dict[str, str]:
    return {
        "career_readiness": "Answer these questions about your professional confidence and skills honestly. There are no right or wrong answers - this is about understanding where you are right now.",
        "work_life_balance": "Answer these questions about managing career and family life. Think about your current situation and typical experiences.",
        "general": "This assessment takes about 5-7 minutes to complete. Answer based on your current situation and feelings, not where you think you should be."
    }

# Validation functions
def validate_responses(responses: Dict[str, Dict[str, str]]) -> tuple[bool, str]:
    if not responses:
        return False, "No responses provided"
    
    if "career_readiness" not in responses or "work_life_balance" not in responses:
        return False, "Both career readiness and work-life balance responses are required"
    
    career_responses = responses.get("career_readiness", {})
    balance_responses = responses.get("work_life_balance", {})
    
    # Check if we have the expected number of questions
    expected_career_questions = len(ASSESSMENT_QUESTIONS["career_readiness"])
    expected_balance_questions = len(ASSESSMENT_QUESTIONS["work_life_balance"])
    
    if len(career_responses) != expected_career_questions:
        return False, f"Expected {expected_career_questions} career readiness responses, got {len(career_responses)}"
    
    if len(balance_responses) != expected_balance_questions:
        return False, f"Expected {expected_balance_questions} work-life balance responses, got {len(balance_responses)}"
    
    # Validate individual responses
    for section_name, section_responses in responses.items():
        if section_name not in ASSESSMENT_QUESTIONS:
            continue
            
        for q_id, response in section_responses.items():
            # Find the question
            question_found = False
            for question in ASSESSMENT_QUESTIONS[section_name]:
                if question["id"] == q_id:
                    question_found = True
                    if response not in question["options"]:
                        return False, f"Invalid response '{response}' for question '{q_id}'"
                    break
            
            if not question_found:
                return False, f"Unknown question ID: '{q_id}'"
    
    return True, "Valid responses"

# Helper function to get question by ID
def get_question_by_id(section: str, question_id: str) -> Dict[str, Any]:
    if section not in ASSESSMENT_QUESTIONS:
        return None
    
    for question in ASSESSMENT_QUESTIONS[section]:
        if question["id"] == question_id:
            return question
    
    return None