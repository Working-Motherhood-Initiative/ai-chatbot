import os
import logging
from typing import List, Dict, Optional
from pathlib import Path
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import openai
import gdown
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class LabourLawRAG:    
    def __init__(self, pdf_directory: str = "labour_laws", openai_api_key: str = None):
        self.pdf_directory = pdf_directory
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.vectorstore = None
        self.embeddings = None
        self.documents_loaded = False
        
        # Google Drive configuration
        self.gdrive_file_id = os.getenv("GDRIVE_FILE_ID")
        self.local_vectorstore_path = "vectorstore/labour_laws.pkl"
        
        self.supported_countries = [
            "Cambodia", "Cameroon", "Congo", "Ethiopia", "Gambia", 
            "Ghana", "Kenya", "Lesotho", "Liberia", "Malawi",
            "Namibia", "Nigeria", "Rwanda", "Sierra Leone", "South Africa",
            "Uganda", "Zambia", "Zimbabwe", "Botswana"
        ]
        
        # Initialize embeddings model (runs locally, no API needed)
        logger.info("Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize OpenAI client
        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        else:
            logger.warning("OpenAI API key not found")
    
    def download_vectorstore_from_gdrive(self) -> bool:
        """
        Download vectorstore PKL file from Google Drive
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.gdrive_file_id:
                logger.warning("GDRIVE_FILE_ID not set - skipping Google Drive download")
                return False
            
            # Check if file already exists locally
            if os.path.exists(self.local_vectorstore_path):
                logger.info(f"Vectorstore file already exists at {self.local_vectorstore_path}")
                return True
            
            logger.info(f"Downloading vectorstore from Google Drive (ID: {self.gdrive_file_id})...")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.local_vectorstore_path), exist_ok=True)
            
            # Try multiple download methods
            success = False
            
            # Method 1: Try gdown with fuzzy option
            try:
                logger.info("Attempting download with gdown (fuzzy mode)...")
                url = f"https://drive.google.com/uc?id={self.gdrive_file_id}"
                gdown.download(url, self.local_vectorstore_path, quiet=False, fuzzy=True)
                
                if os.path.exists(self.local_vectorstore_path) and os.path.getsize(self.local_vectorstore_path) > 1000:
                    success = True
            except Exception as e:
                logger.warning(f"gdown fuzzy mode failed: {e}")
            
            # Method 2: Try direct download URL
            if not success:
                try:
                    logger.info("Attempting direct download...")
                    import requests
                    
                    # Use the direct download link format
                    download_url = f"https://drive.google.com/uc?export=download&id={self.gdrive_file_id}"
                    
                    session = requests.Session()
                    response = session.get(download_url, stream=True)
                    
                    # Handle Google Drive virus scan warning
                    for key, value in response.cookies.items():
                        if key.startswith('download_warning'):
                            params = {'id': self.gdrive_file_id, 'confirm': value}
                            response = session.get(download_url, params=params, stream=True)
                            break
                    
                    # Save file
                    with open(self.local_vectorstore_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    if os.path.exists(self.local_vectorstore_path) and os.path.getsize(self.local_vectorstore_path) > 1000:
                        success = True
                        logger.info("Direct download successful")
                except Exception as e:
                    logger.warning(f"Direct download failed: {e}")
            
            # Verify file exists and is readable
            if success and os.path.exists(self.local_vectorstore_path):
                file_size = os.path.getsize(self.local_vectorstore_path) / (1024 * 1024)
                logger.info(f"Successfully downloaded vectorstore ({file_size:.2f} MB)")
                return True
            else:
                logger.error("All download methods failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to download vectorstore from Google Drive: {e}")
            return False
    
    def load_vectorstore_from_file(self, file_path: str) -> bool:
        """
        Load vectorstore from PKL file
        
        Args:
            file_path: Path to the PKL file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Loading vectorstore from {file_path}...")
            
            if not os.path.exists(file_path):
                logger.error(f"Vectorstore file not found: {file_path}")
                return False
            
            with open(file_path, 'rb') as f:
                self.vectorstore = pickle.load(f)
            
            self.documents_loaded = True
            doc_count = len(self.vectorstore.docstore._dict)
            logger.info(f"Successfully loaded vectorstore with {doc_count} document chunks")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vectorstore from file: {e}")
            return False
    
    def load_and_process_documents(self, force_reload: bool = False, use_gdrive: bool = True) -> int:
        """
        Load vectorstore either from Google Drive or local PDFs
        
        Args:
            force_reload: Force reload from PDFs (ignores existing vectorstore)
            use_gdrive: Try to load from Google Drive first
            
        Returns:
            int: Number of document chunks loaded
        """
        
        # PRIORITY 1: Try loading from Google Drive (fastest)
        if use_gdrive and not force_reload:
            logger.info("Attempting to load vectorstore from Google Drive...")
            
            # Download from Google Drive if needed
            download_success = self.download_vectorstore_from_gdrive()
            
            if download_success:
                # Try to load the downloaded file
                load_success = self.load_vectorstore_from_file(self.local_vectorstore_path)
                
                if load_success:
                    logger.info("Successfully loaded vectorstore from Google Drive")
                    return len(self.vectorstore.docstore._dict)
                else:
                    logger.warning("Downloaded file but failed to load - will try other methods")
        
        # PRIORITY 2: Try loading from existing local file
        if os.path.exists(self.local_vectorstore_path) and not force_reload:
            logger.info("Attempting to load existing local vectorstore...")
            
            load_success = self.load_vectorstore_from_file(self.local_vectorstore_path)
            
            if load_success:
                logger.info("Successfully loaded existing local vectorstore")
                return len(self.vectorstore.docstore._dict)
        
        # PRIORITY 3: Load from PDFs (slowest, fallback option)
        logger.info("Loading vectorstore from PDFs (this may take a while)...")
        return self._load_from_pdfs()
    
    def _load_from_pdfs(self) -> int:
        """
        Load and process documents from PDF files (original method)
        
        Returns:
            int: Number of document chunks created
        """
        logger.info(f"Loading labour law PDFs from {self.pdf_directory}...")
        
        if not os.path.exists(self.pdf_directory):
            logger.error(f"PDF directory not found: {self.pdf_directory}")
            raise FileNotFoundError(f"Labour law directory not found: {self.pdf_directory}")
        
        all_documents = []
        pdf_files = list(Path(self.pdf_directory).glob("*.pdf"))
        
        if not pdf_files:
            logger.error(f"No PDF files found in {self.pdf_directory}")
            raise FileNotFoundError(f"No labour law PDFs found in {self.pdf_directory}")
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            try:
                logger.info(f"Processing {pdf_path.name}...")
                
                # Extract country name from filename
                country = self._extract_country_from_filename(pdf_path.name)
                
                # Load PDF
                loader = PyPDFLoader(str(pdf_path))
                documents = loader.load()
                
                # Add metadata
                for doc in documents:
                    doc.metadata['source'] = pdf_path.name
                    doc.metadata['country'] = country
                
                all_documents.extend(documents)
                logger.info(f"Loaded {len(documents)} pages from {pdf_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
                continue
        
        if not all_documents:
            raise ValueError("No documents were successfully loaded")
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        
        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        splits = text_splitter.split_documents(all_documents)
        logger.info(f"Created {len(splits)} text chunks")
        
        # Create vectorstore
        logger.info("Creating vectorstore with embeddings...")
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        self.documents_loaded = True
        
        # Save vectorstore locally
        logger.info("Saving vectorstore...")
        os.makedirs(os.path.dirname(self.local_vectorstore_path), exist_ok=True)
        with open(self.local_vectorstore_path, 'wb') as f:
            pickle.dump(self.vectorstore, f)
        
        logger.info(f"Successfully loaded and processed {len(splits)} document chunks")
        logger.info(f"Vectorstore saved to {self.local_vectorstore_path}")
        logger.info("TIP: Upload this file to Google Drive and set GDRIVE_FILE_ID for faster loading")
        
        return len(splits)
    
    def _extract_country_from_filename(self, filename: str) -> str:
        """Extract country name from PDF filename"""
        filename_lower = filename.lower()
        
        for country in self.supported_countries:
            if country.lower() in filename_lower:
                return country
        
        return "Unknown"
    
    def search_relevant_documents(
        self, 
        query: str, 
        country: Optional[str] = None,
        top_k: int = 4
    ) -> List[Document]:
        if not self.documents_loaded or not self.vectorstore:
            raise ValueError("Documents not loaded. Call load_and_process_documents() first.")
        
        # Search with similarity
        if country:
            # Filter by country in metadata
            search_kwargs = {
                "k": top_k * 2,
                "filter": {"country": country}
            }
            try:
                docs = self.vectorstore.similarity_search(query, **search_kwargs)
                if not docs:
                    logger.warning(f"No results for {country}, searching all countries")
                    docs = self.vectorstore.similarity_search(query, k=top_k)
            except Exception as e:
                logger.warning(f"Filtered search failed: {e}, using unfiltered search")
                docs = self.vectorstore.similarity_search(query, k=top_k)
        else:
            docs = self.vectorstore.similarity_search(query, k=top_k)
        
        return docs[:top_k]
    
    def generate_answer(
        self, 
        query: str, 
        country: Optional[str] = None,
        user_context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict:
        if not self.documents_loaded:
            raise ValueError("Documents not loaded. Call load_and_process_documents() first.")
        
        logger.info(f"Searching for: {query}" + (f" in {country}" if country else ""))
        relevant_docs = self.search_relevant_documents(query, country)

        if not relevant_docs:
            return {
                "answer": "I couldn't find specific information about that in the labour law documents. Could you rephrase your question or specify a country?",
                "sources": [],
                "country": country,
                "confidence": "low",
                "chat_history": chat_history or []
            }

        # Prepare RAG context
        context = "\n\n".join([
            f"[Source: {doc.metadata.get('country', 'Unknown')} - {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc in relevant_docs
        ])

        system_prompt = """You are a helpful labour law assistant specializing in African countries.
You provide accurate, clear, and supportive answers to working mothers about their labour rights.

Guidelines:
- Base your answer ONLY on the provided context from official labour law documents
- Be clear, supportive, and practical in your advice
- If information is specific to certain countries, state this clearly
- If the context doesn't contain enough information, say so honestly
- Focus on rights related to maternity leave, workplace discrimination, flexible work, and career protection
- Use simple, accessible language
"""

        # Construct messages for OpenAI chat API
        messages = [{"role": "system", "content": system_prompt}]

        # Limit history to last 8 turns
        if chat_history:
            truncated = chat_history[-8:]
            messages.extend(truncated)

        # Add the current question with context
        user_prompt = f"""Question: {query}

{f"User Context: {user_context}" if user_context else ""}

Labour Law Context:
{context}

Please provide a clear, accurate answer based ONLY on the information above. Structure your response with:
1. Direct answer to the question
2. Relevant legal provisions or rights
3. Practical advice for working mothers
4. Any country-specific differences if applicable

If the context doesn't contain the answer, say so clearly."""
        
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3,
                max_tokens=800
            )

            answer = response.choices[0].message.content.strip()

            # Extract sources
            sources = []
            seen_sources = set()
            for doc in relevant_docs:
                source_key = f"{doc.metadata.get('country', 'Unknown')}_{doc.metadata.get('source', 'Unknown')}"
                if source_key not in seen_sources:
                    sources.append({
                        "country": doc.metadata.get('country', 'Unknown'),
                        "document": doc.metadata.get('source', 'Unknown'),
                        "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    })
                    seen_sources.add(source_key)

            # Update chat history
            updated_history = (chat_history or [])[-8:]
            updated_history.append({"role": "user", "content": query})
            updated_history.append({"role": "assistant", "content": answer})

            return {
                "answer": answer,
                "sources": sources,
                "country": country if country else "Multiple countries",
                "confidence": "high" if len(relevant_docs) >= 3 else "medium",
                "query": query,
                "chat_history": updated_history
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": "I encountered an error processing your question. Please try rephrasing it.",
                "sources": [],
                "country": country,
                "confidence": "error",
                "error": str(e),
                "chat_history": chat_history or []
            }
    
    def reload_from_gdrive(self) -> int:
        """
        Force reload vectorstore from Google Drive
        Useful for manual updates
        
        Returns:
            int: Number of document chunks loaded
        """
        logger.info("Force reloading vectorstore from Google Drive...")
        
        # Remove local cache
        if os.path.exists(self.local_vectorstore_path):
            os.remove(self.local_vectorstore_path)
            logger.info("Removed cached vectorstore file")
        
        # Download fresh copy
        download_success = self.download_vectorstore_from_gdrive()
        
        if not download_success:
            raise Exception("Failed to download vectorstore from Google Drive")
        
        # Load the new file
        load_success = self.load_vectorstore_from_file(self.local_vectorstore_path)
        
        if not load_success:
            raise Exception("Failed to load downloaded vectorstore")
        
        logger.info("Successfully reloaded vectorstore from Google Drive")
        return len(self.vectorstore.docstore._dict)
    
    def get_supported_countries(self) -> List[str]:
        """Get list of supported countries"""
        return self.supported_countries
    
    def get_system_stats(self) -> Dict:
        if not self.vectorstore:
            return {
                "documents_loaded": False,
                "total_chunks": 0,
                "supported_countries": len(self.supported_countries),
                "gdrive_enabled": bool(self.gdrive_file_id)
            }
        
        return {
            "documents_loaded": self.documents_loaded,
            "total_chunks": len(self.vectorstore.docstore._dict) if self.vectorstore else 0,
            "supported_countries": len(self.supported_countries),
            "countries": self.supported_countries,
            "gdrive_enabled": bool(self.gdrive_file_id),
            "local_cache_exists": os.path.exists(self.local_vectorstore_path)
        }


# Global instance
_rag_instance = None

def get_rag_instance() -> LabourLawRAG:
    """Get or create global RAG instance"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = LabourLawRAG()
    return _rag_instance

def initialize_rag_system(
    pdf_directory: str = "labour_laws", 
    force_reload: bool = False,
    use_gdrive: bool = True
) -> int:
    """
    Initialize the RAG system
    
    Args:
        pdf_directory: Path to PDF directory (fallback)
        force_reload: Force reload from PDFs
        use_gdrive: Try to load from Google Drive first
        
    Returns:
        int: Number of document chunks loaded
    """
    rag = get_rag_instance()
    return rag.load_and_process_documents(force_reload=force_reload, use_gdrive=use_gdrive)