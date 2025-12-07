import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict

logger = logging.getLogger(__name__)

class SessionStore:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
        
    def create_session(self) -> str:
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            "created": datetime.now(),
            "expires": datetime.now() + timedelta(hours=2),
            "requests": 0
        }
        logger.info(f"Created new session: {session_id[:8]}...")
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        if session["expires"] < datetime.now():
            del self.sessions[session_id]
            return False
        
        session["requests"] += 1
        return True
    
    def cleanup_expired(self):
        now = datetime.now()
        expired = [sid for sid, data in self.sessions.items() 
                  if data["expires"] < now]
        for sid in expired:
            del self.sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    def get_stats(self):
        return {
            "active_sessions": len(self.sessions),
            "total_sessions": len(self.sessions)
        }