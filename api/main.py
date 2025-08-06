from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict, Any, Union
import os
import uuid
import shutil
import tempfile
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import sys
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
        RotatingFileHandler('api.log', maxBytes=10485760, backupCount=5)  # 10MB per file, 5 files max
    ]
)
logger = logging.getLogger(__name__)

# Also log all uvicorn logs
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(logging.DEBUG)
uvicorn_access = logging.getLogger("uvicorn.access")
uvicorn_access.setLevel(logging.DEBUG)

client = chromadb.PersistentClient(path="chroma_store")
sentence_transformer_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Import your existing pipeline components

from INGESTION.downloader import download_file
from INGESTION.parser import read_document
from INGESTION.chunker import split_text as chunk_text
from EMBEDDING.process import process_document
from LLM import rag_query, generate_response
from SEARCHING.semantic_search import semantic_search, get_context_with_sources

# Import LLM module if available, otherwise use a placeholder
try:
    from LLM.query import get_answer
except ImportError:
    # Fallback implementation if LLM module is not available
    def get_answer(question: str, context: str) -> str:
        return f"Answer to '{question}': [Simulated response based on context]"

app = FastAPI(
    title="RAG API",
    version="1.0.0",
    description="Retrieval-Augmented Generation API for document processing and question-answering"
)

# Security
security = HTTPBearer()
API_TOKENS = ["c742772b47bb55597517747abafcc3d472fa1c4403a1574461aa3f70ea2d9301"]

# Configuration
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Models
class RunRequest(BaseModel):
    documents: HttpUrl = Field(..., description="URL of the document to process")
    questions: List[str] = Field(..., min_items=1, description="List of questions to answer")
    conversation_id: Optional[str] = Field(
        None, 
        description="Optional conversation ID for maintaining context across multiple interactions"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "documents": "https://example.com/document.pdf",
                "questions": ["What is this document about?"],
                "conversation_id": "optional-conversation-id"
            }
        }

class RunResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the questions")
    sources: List[str] = Field(..., description="List of source documents used for answering")
    conversation_id: Optional[str] = Field(
        None,
        description="Conversation ID for maintaining context"
    )

# Authentication
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials not in API_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Utility functions
def get_temp_dir() -> str:
    """Create and return a path to a temporary directory"""
    temp_dir = os.path.join(tempfile.gettempdir(), "rag_processing")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def cleanup_directory(directory: str):
    """Safely remove a directory and its contents"""
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Could not clean up directory {directory}: {e}")

# API Endpoints
@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def run_pipeline(
    request: RunRequest,
    token: str = Depends(verify_token)
):
    """
    Process a document from a URL and answer questions about its content.
    
    - **documents**: URL of the document to process (PDF, DOCX, or TXT)
    - **questions**: List of questions to answer about the document
    - **conversation_id**: Optional ID for maintaining conversation context
    """
    temp_dir = None
    try:
        # Create a temporary directory for processing
        temp_dir = get_temp_dir()
        
        # 1. Download the document
        file_path, content_type = download_file(request.documents, str(temp_dir))
        
        # 2. Parse the document
        logger.info(f"Reading document content from: {file_path}")
        try:
            text_content = read_document(file_path)
            if not text_content or not text_content.strip():
                raise ValueError("Document is empty or could not be read")
            logger.info(f"Successfully read document with {len(text_content)} characters")
        except Exception as e:
            logger.error(f"Error reading document: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read document: {str(e)}"
            )
        
        # 3. Process and embed the document (chunking happens inside process_document)
        collection_name = f"doc_{uuid.uuid4().hex}"
        logger.info(f"Processing and embedding document in collection: {collection_name}")
        try:
            process_document(
                text_content=text_content,
                collection_name=collection_name,
                metadata={"source": str(request.documents)}
            )
            logger.info("Document processed and embedded successfully")
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing document: {str(e)}"
            )
        
        # Get the collection for querying
        try:
            collection = client.get_collection(
                name=collection_name,
                embedding_function=sentence_transformer_ef
            )
            if collection is None:
                raise ValueError("Failed to get collection after processing document")
            logger.info("Successfully retrieved collection for querying")
        except Exception as e:
            logger.error(f"Error getting collection: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error accessing document collection: {str(e)}"
            )
        
        # 5. Answer each question
        answers = []
        all_sources = set()
        
        for question in request.questions:
            answer, sources = rag_query(
                collection=collection,
                query=question,
                n_chunks=3,  # Get top 3 most relevant chunks
                conversation_history=request.conversation_id or ""
            )
            answers.append(answer)
            all_sources.update(sources)
        
        # Clean up the collection after processing
        client.delete_collection(collection_name)
        
        return {
            "answers": answers,
            "sources": list(all_sources),
            "conversation_id": request.conversation_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )
    finally:
        # Clean up temporary files
        if temp_dir and os.path.exists(temp_dir):
            cleanup_directory(temp_dir)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.on_event("startup")
async def startup_event():
    """Initialize services when the application starts."""
    logger.info("Starting up FastAPI application...")
    # Initialize any required services here

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up services when the application shuts down."""
    logger.info("Shutting down FastAPI application...")
    # Clean up resources here

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
