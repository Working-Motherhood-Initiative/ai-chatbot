import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def get_ai_response(openai_client, messages: List[Dict], max_tokens: int = 500) -> str:
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