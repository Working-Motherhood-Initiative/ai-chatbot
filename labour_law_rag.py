import os
import logging
from typing import List, Dict, Optional
from pathlib import Path
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import openai
import gdown
import requests
from dotenv import load_dotenv
import re

load_dotenv()

logger = logging.getLogger(__name__)

class LabourLawRAG:    
    def __init__(self, pdf_directory: str = "labour_laws", openai_api_key: str = None):
        self.pdf_directory = pdf_directory
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.vectorstore = None
        self.embeddings = None
        self.documents_loaded = False
        self.gdrive_file_id = os.getenv("GDRIVE_FILE_ID")
        self.local_vectorstore_path = "vectorstore/labour_laws.pkl"
        self.supported_countries = [
            "Cambodia", "Cameroon", "Congo", "Ethiopia", "Gambia", 
            "Ghana", "Kenya", "Lesotho", "Liberia", "Malawi",
            "Namibia", "Nigeria", "Rwanda", "Sierra Leone", "South Africa",
            "Uganda", "Zambia", "Zimbabwe", "Botswana"
        ]
        
        logger.info("Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        else:
            logger.warning("OpenAI API key not found")
    
    def download_vectorstore_from_gdrive(self) -> bool:
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
            
            success = False
            
            try:
                logger.info("Attempting download with gdown (fuzzy mode)...")
                url = f"https://drive.google.com/uc?id={self.gdrive_file_id}"
                gdown.download(url, self.local_vectorstore_path, quiet=False, fuzzy=True)
                
                if os.path.exists(self.local_vectorstore_path) and os.path.getsize(self.local_vectorstore_path) > 1000:
                    success = True
            except Exception as e:
                logger.warning(f"gdown fuzzy mode failed: {e}")
            
            if not success:
                try:
                    logger.info("Attempting direct download...")
                    
                    download_url = f"https://drive.google.com/uc?export=download&id={self.gdrive_file_id}"
                    
                    session = requests.Session()
                    response = session.get(download_url, stream=True)
                    
                    for key, value in response.cookies.items():
                        if key.startswith('download_warning'):
                            params = {'id': self.gdrive_file_id, 'confirm': value}
                            response = session.get(download_url, params=params, stream=True)
                            break
                    
                    with open(self.local_vectorstore_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    if os.path.exists(self.local_vectorstore_path) and os.path.getsize(self.local_vectorstore_path) > 1000:
                        success = True
                        logger.info("Direct download successful")
                except Exception as e:
                    logger.warning(f"Direct download failed: {e}")
            
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
        
        if os.path.exists(self.local_vectorstore_path) and not force_reload:
            logger.info("Attempting to load existing local vectorstore...")
            
            load_success = self.load_vectorstore_from_file(self.local_vectorstore_path)
            
            if load_success:
                logger.info("Successfully loaded existing local vectorstore")
                return len(self.vectorstore.docstore._dict)
        
        logger.info("Loading vectorstore from PDFs (this may take a while)...")
        return self._load_from_pdfs()
    
    def _load_from_pdfs(self) -> int:
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
                
                loader = PyPDFLoader(str(pdf_path))
                documents = loader.load()
                
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
        
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        splits = text_splitter.split_documents(all_documents)
        logger.info(f"Created {len(splits)} text chunks")
        
        logger.info("Creating vectorstore with embeddings...")
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        self.documents_loaded = True
        
        logger.info("Saving vectorstore...")
        os.makedirs(os.path.dirname(self.local_vectorstore_path), exist_ok=True)
        with open(self.local_vectorstore_path, 'wb') as f:
            pickle.dump(self.vectorstore, f)
        
        logger.info(f"Successfully loaded and processed {len(splits)} document chunks")
        logger.info(f"Vectorstore saved to {self.local_vectorstore_path}")
        logger.info("TIP: Upload this file to Google Drive and set GDRIVE_FILE_ID for faster loading")
        
        return len(splits)
    
    def _extract_country_from_filename(self, filename: str) -> str:
        filename_lower = filename.lower()
        
        for country in self.supported_countries:
            if country.lower() in filename_lower:
                return country
        
        return "Unknown"
    
    def search_relevant_documents(
        self, 
        query: str, 
        country: Optional[str] = None,
        top_k: int = 6
    ) -> List[Document]:
        """
        Search for relevant documents, optionally filtered by country
        
        Args:
            query: The search query
            country: Country name to filter by (e.g., "Ghana")
            top_k: Number of top results to return
        """
        if not self.documents_loaded or not self.vectorstore:
            raise ValueError("Documents not loaded. Call load_and_process_documents() first.")
        
        # Search with similarity
        if country:
            # Search with increased k to ensure we get enough results for the country
            search_k = top_k * 8
            
            try:
                # Get all results first
                all_docs = self.vectorstore.similarity_search(query, k=search_k)
                
                # Filter for selected country (case-insensitive)
                country_lower = country.lower()
                filtered_docs = [
                    doc for doc in all_docs 
                    if doc.metadata.get('country', '').lower() == country_lower
                ]
                
                if not filtered_docs:
                    logger.warning(f"No results for {country}, searching all countries")
                    docs = self.vectorstore.similarity_search(query, k=top_k)
                else:
                    docs = filtered_docs[:top_k]
                    
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
        """Generate an answer using RAG with proper formatting"""
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
- When asked "who do I report to" or similar questions, ALWAYS identify the specific authority mentioned in the context.Mention the specific institutions full name.
- Always cite specific section numbers from the law (e.g., "Section 57" or "Section 55(2)")
- Be specific with numbers and timeframes - use exact terms from the law (e.g., "at least 12 weeks" not "several weeks")
- If the context mentions complaint procedures, state them clearly and step-by-step
- Be clear, supportive, and practical in your advice
- Focus on actionable information - what the person should do, where to go, what documentation to bring
- Use simple, accessible language
- If the context doesn't contain enough information, say so honestly
- DO NOT hallucinate or make up answers.
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

Please provide a clear, accurate answer based ONLY on the information above.

CRITICAL REQUIREMENTS:
1. If the question asks "who do I report to" or "where do I complain", you MUST identify the specific authority/commission/body mentioned in the context (e.g., National Labour Commission, Chief Labour Officer, Minister, etc.)
2. Always cite specific section numbers when referencing legal provisions (e.g., "Section 57" or "Section 55(2)")
3. If the context mentions complaint procedures, explicitly state them
4. Be specific about rights - use exact terms from the law (e.g., "at least 12 weeks" not "several weeks")

Structure your response naturally:
1. Start with a direct answer identifying the specific reporting authority if asked
2. List the relevant legal rights with section numbers
3. Provide practical step-by-step advice for taking action
4. Note any important protections or consequences

Important: Write naturally and conversationally. Use proper paragraphs and bullet points where appropriate. Do NOT include section labels or headers in your response - just provide the information in a flowing, readable format.

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
            
            # Apply enhanced formatting cleanup
            answer = self._clean_answer_formatting(answer)

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

    def _clean_answer_formatting(self, answer: str) -> str:
        """Clean and format the answer with proper spacing"""
        # Replace escaped newlines
        answer = answer.replace('\\n', '\n')
        
        # Remove common formatting artifacts from GPT responses
        # Remove markdown code blocks
        answer = re.sub(r'```json\s*', '', answer)
        answer = re.sub(r'```\s*', '', answer)
        
        # Remove explicit section headers that GPT might add
        answer = re.sub(r'\*\*Direct Answer:\*\*\s*', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\*\*Legal Provisions:\*\*\s*', '\n\n', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\*\*Practical Advice:\*\*\s*', '\n\n', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\*\*Country-Specific Notes:\*\*\s*', '\n\n', answer, flags=re.IGNORECASE)
        answer = re.sub(r'Direct Answer:\s*', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'Legal Provisions:\s*', '\n\n', answer, flags=re.IGNORECASE)
        answer = re.sub(r'Practical Advice:\s*', '\n\n', answer, flags=re.IGNORECASE)
        answer = re.sub(r'Country-Specific Notes:\s*', '\n\n', answer, flags=re.IGNORECASE)
        
        answer = re.sub(r'([^\n])(\n)(\d+\.)', r'\1\n\n\3', answer)
        
        answer = re.sub(r'([^\n])(\n)([-•*])', r'\1\n\n\3', answer)
        
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        
        answer = re.sub(r'[ \t]+\n', '\n', answer)
        
        lines = answer.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip().startswith(('•', '-', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                cleaned_lines.append(line.strip())
            else:
                cleaned_lines.append(line.strip())
        answer = '\n'.join(cleaned_lines)
        
        answer = re.sub(r'([-•*])([^\s])', r'\1 \2', answer)
        
        answer = answer.strip('"\'')
        
        return answer.strip()
    
    def reload_from_gdrive(self) -> int:
        logger.info("Force reloading vectorstore from Google Drive...")
        
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
        """Return list of supported countries"""
        return self.supported_countries
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
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
    """Initialize the RAG system"""
    rag = get_rag_instance()
    return rag.load_and_process_documents(force_reload=force_reload, use_gdrive=use_gdrive)