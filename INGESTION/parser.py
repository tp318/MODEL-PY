import os
import logging
import docx
import PyPDF2
from typing import Optional

# Set up logging
logger = logging.getLogger(__name__)

def read_text_file(file_path: str) -> str:
    """Read content from a plain text (.txt) file."""
    try:
        logger.info(f"Reading text file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            if not content.strip():
                logger.warning(f"Text file appears to be empty: {file_path}")
            return content
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode text file {file_path} with UTF-8 encoding: {e}")
        # Try with different encodings as fallback
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as fallback_e:
            logger.error(f"Failed to read text file with fallback encoding: {fallback_e}")
            raise ValueError(f"Failed to read text file: {str(fallback_e)}")
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}", exc_info=True)
        raise ValueError(f"Error reading text file: {str(e)}")

def read_pdf_file(file_path: str) -> str:
    """Read content from a PDF file."""
    logger.info(f"Reading PDF file: {file_path}")
    text = ""
    try:
        with open(file_path, 'rb') as file:
            try:
                pdf_reader = PyPDF2.PdfReader(file)
                if not pdf_reader.pages:
                    logger.warning(f"PDF file contains no pages: {file_path}")
                    return ""
                
                for i, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as page_e:
                        logger.warning(f"Error extracting text from page {i} of {file_path}: {page_e}")
                        continue
                
                if not text.strip():
                    logger.warning(f"No extractable text found in PDF: {file_path}")
                
                return text
                
            except PyPDF2.errors.PdfReadError as e:
                logger.error(f"Error reading PDF {file_path}: {e}")
                raise ValueError(f"Invalid or corrupted PDF file: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error reading PDF file {file_path}: {e}", exc_info=True)
        raise ValueError(f"Failed to read PDF: {str(e)}")

def read_docx_file(file_path: str) -> str:
    """Read content from a Word (.docx) file."""
    logger.info(f"Reading DOCX file: {file_path}")
    try:
        doc = docx.Document(file_path)
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        
        if not paragraphs:
            logger.warning(f"No readable text content found in DOCX: {file_path}")
            
        return "\n".join(paragraphs)
        
    except Exception as e:
        logger.error(f"Error reading DOCX file {file_path}: {e}", exc_info=True)
        raise ValueError(f"Failed to read Word document: {str(e)}")

def read_document(file_path: str) -> str:
    """
    Dispatch function to read a document based on file extension.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        str: Extracted text content from the document
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file type is unsupported or there's an error reading the file
    """
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    if not os.path.isfile(file_path):
        error_msg = f"Path is not a file: {file_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        _, ext = os.path.splitext(file_path.lower())
        logger.info(f"Processing file with extension: {ext}")
        
        if ext == ".txt":
            return read_text_file(file_path)
        elif ext == ".pdf":
            return read_pdf_file(file_path)
        elif ext in (".docx", ".doc"):
            return read_docx_file(file_path)
        else:
            error_msg = f"Unsupported file type: {ext}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    except Exception as e:
        logger.error(f"Error in read_document for {file_path}: {e}", exc_info=True)
        raise ValueError(f"Failed to process document: {str(e)}")