import os
import re
from typing import Optional, List

# === Get latest doc[N].* file ===
def get_latest_doc_file():
    files = os.listdir('.')
    pattern = r'doc\[(\d+)\]\.(pdf|docx|txt)'
    matched = [(int(re.findall(pattern, f)[0][0]), f) for f in files if re.findall(pattern, f)]
    if not matched:
        raise FileNotFoundError("No doc[N].* files found.")
    latest_file = sorted(matched, key=lambda x: x[0], reverse=True)[0][1]
    return latest_file

# === Read document dispatcher ===
def read_document(file_path: str) -> str:
    """
    Read document content from file path.
    
    Args:
        file_path: Path to the document file (PDF, DOCX, or TXT)
        
    Returns:
        str: Extracted text content from the document
        
    Raises:
        ValueError: If the file format is not supported
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if file_path.endswith(".pdf"):
        return read_pdf_file(file_path)
    elif file_path.endswith(".docx"):
        return read_docx_file(file_path)
    elif file_path.endswith(".txt"):
        return read_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {os.path.splitext(file_path)[1]}")

# === Split text into sentence-based chunks ===
def split_text(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Split text into chunks of approximately chunk_size characters,
    trying to break at sentence boundaries.
    
    Args:
        text: The input text to be chunked
        chunk_size: Target size for each chunk in characters
        
    Returns:
        List[str]: List of text chunks
    """
    if not text or not isinstance(text, str):
        return []
        
    # Normalize whitespace and split into sentences
    text = ' '.join(text.split())
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Ensure sentence ends with punctuation
        if not re.search(r'[.!?]$', sentence):
            sentence += '.'
            
        sentence_size = len(sentence)
        
        # If current chunk is empty, always add at least one sentence
        if not current_chunk:
            current_chunk.append(sentence)
            current_size = sentence_size
        # If adding this sentence would exceed chunk size, finalize current chunk
        elif current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for the space

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# === Helper functions for document reading ===

def read_pdf_file(file_path: str) -> str:
    """Read text content from a PDF file."""
    import PyPDF2
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {str(e)}")
    return text

def read_docx_file(file_path: str) -> str:
    """Read text content from a DOCX file."""
    import docx
    try:
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        raise ValueError(f"Error reading DOCX file: {str(e)}")

def read_text_file(file_path: str) -> str:
    """Read text content from a plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        raise ValueError(f"Error reading text file: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        try:
            text = read_document(file_path)
            chunks = split_text(text)
            print(f"Split document into {len(chunks)} chunks")
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks as example
                print(f"\nChunk {i+1} (length: {len(chunk)}):")
                print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("Usage: python -m INGESTION.chunker <file_path>")