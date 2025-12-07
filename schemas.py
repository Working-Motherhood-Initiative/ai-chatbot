from pydantic import BaseModel
from typing import Optional, List, Dict

class LabourLawQuery(BaseModel):
    query: str
    country: Optional[str] = None
    user_context: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None

class AssessmentResponse(BaseModel):
    career_readiness: Dict[str, str]
    work_life_balance: Dict[str, str]