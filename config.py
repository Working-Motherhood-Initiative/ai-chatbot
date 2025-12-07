import os
import logging
import threading
import asyncio
from dotenv import load_dotenv
import openai
from job_fetcher import preload_job_embeddings, get_all_jobs
from labour_law_rag import initialize_rag_system, get_rag_instance

logger = logging.getLogger(__name__)

_initialization_complete = False
_initialization_error = None

def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(level=getattr(logging, log_level))
    return logging.getLogger(__name__)

def load_environment():
    load_dotenv()
    api_token = os.getenv("API_TOKEN")
    if not api_token:
        raise Exception("API_TOKEN not found in environment variables")
    return api_token

def initialize_openai_client():
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("OpenAI client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise

def initialize_background():
    global _initialization_complete, _initialization_error
    try:
        logger.info("Background initialization started...")
        
        logger.info("Starting job data initialization...")
        preload_job_embeddings()
        jobs = get_all_jobs()
        logger.info(f"Successfully loaded {len(jobs)} jobs")
        
        logger.info("Initializing Labour Law RAG system...")
        try:
            doc_count = initialize_rag_system(
                pdf_directory="labour_laws",
                force_reload=False,
                use_gdrive=True
            )
            logger.info(f"Successfully loaded {doc_count} labour law document chunks")
            
            rag = get_rag_instance()
            stats = rag.get_system_stats()
            logger.info(f"Google Drive enabled: {stats.get('gdrive_enabled', False)}")
            logger.info(f"Local cache exists: {stats.get('local_cache_exists', False)}")
            
        except Exception as e:
            logger.error(f"Labour Law RAG initialization failed: {e}")
            logger.warning("Labour law queries will not be available until next restart")
            _initialization_error = str(e)
        
        _initialization_complete = True
        logger.info("Background initialization completed!")
        
    except Exception as e:
        logger.error(f"Background initialization error: {e}")
        _initialization_error = str(e)
        _initialization_complete = True

async def cleanup_sessions_periodically(session_store):
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        session_store.cleanup_expired()
        logger.info(f"Session stats: {session_store.get_stats()}")

def test_privacy_protection(privacy_func):
    test_text = "John Doe john@email.com (555) 123-4567 Professional Summary: Software developer"
    cleaned = privacy_func(test_text)
    if "john@email.com" in cleaned:
        raise Exception("Privacy protection system failed startup test")
    logger.info("Privacy protection startup test passed")

def start_background_initialization():
    init_thread = threading.Thread(target=initialize_background, daemon=True)
    init_thread.start()
    logger.info("Background initialization thread started")