import os
import chromadb
import logging
import torch
from typing import Optional, Dict
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import gc
import re
import nltk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize the sentence transformer model
# Use L6-v2 for better accuracy — it's worth the small speed cost
model = SentenceTransformer("all-MiniLM-L6-v2")

# Move model to GPU if available
if torch.cuda.is_available():
    logger.info("CUDA GPU detected. Moving model to GPU...")
    model.to('cuda')
else:
    logger.warning("CUDA not available. Running on CPU — may be slower.")

# Use Chroma's built-in embedding function (safe and consistent)
sentence_transformer_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="chroma_store")


def _split_into_sections(text: str):
    """
    Split text into sections using policy-specific headers.
    Preserves context like '4. EXCLUSIONS', 'List I', etc.
    """
    header_pattern = r'\n\s*(\d+(?:\.\d+)*[^\n]+|Annexure [IVX]+|[Ll]ist [IVX]+|[Ll]ist of|Table of Benefits|Claims Procedure|4\. EXCLUSIONS|90 Days Waiting Period|iii\. Two years waiting period)'
    parts = []
    start = 0
    for match in re.finditer(header_pattern, text):
        if parts:
            end = match.start()
            content = text[start:end].strip()
            parts[-1]['content'] = content
        header = match.group(1).strip()
        parts.append({'header': header, 'content': ''})
        start = match.start()
    final_content = text[start:].strip()
    if parts:
        parts[-1]['content'] = final_content
    elif final_content:
        parts.append({'header': 'Uncategorized', 'content': final_content})
    return [p for p in parts if p['content'].strip()]


def _split_text_with_overlap(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Generator: Split text into chunks with overlap, using NLTK for sentence boundary detection.
    Preserves tables if detected.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return

    # If it's a table (contains pipes or known table keywords), yield as one block
    if '|' in text[:300] and text.count('|') > 3:
        yield text.strip()
        return

    # Use NLTK for accurate sentence splitting
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        logger.warning(f"NLTK sentence tokenize failed: {e}, falling back to line split")
        sentences = [s.strip() for s in re.split(r'\n+', text) if s.strip()]

    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_len = len(sentence)

        if current_length + sentence_len > chunk_size and current_chunk:
            chunk = ' '.join(current_chunk)
            yield chunk
            # Overlap: carry forward last sentence
            if chunk_overlap > 0:
                last_sentence = current_chunk[-1]
                current_chunk = [last_sentence] if len(last_sentence) < chunk_size else []
                current_length = len(' '.join(current_chunk))
            else:
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += sentence_len
        else:
            current_chunk.append(sentence)
            current_length += sentence_len

    if current_chunk:
        yield ' '.join(current_chunk)


def process_document(
    text_content: str,
    collection_name: str,
    metadata: Optional[Dict[str, str]] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 10  # Keep smaller batch for stability
) -> int:
    """
    Process a document with section-aware, table-preserving, streaming chunking.
    Returns number of chunks processed.
    """
    if not text_content or not isinstance(text_content, str) or not text_content.strip():
        raise ValueError("Invalid or empty text content provided")

    if metadata is None:
        metadata = {}

    logger.info(f"Starting processing of document with {len(text_content):,} characters")

    try:
        # Get or create collection
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef  # Use built-in EF
        )

        # Split into sections to preserve context
        sections = _split_into_sections(text_content)
        total_chunks = 0
        chunk_id_counter = 0

        # Unified generator: yields (chunk, section, type) across all sections
        def unified_chunk_stream():
            for sec in sections:
                header = sec['header']
                content = sec['content']
                if len(content.strip()) < 50:
                    continue
                is_table = '|' in content[:300] and content.count('|') > 3
                for chunk in _split_text_with_overlap(content, chunk_size, chunk_overlap):
                    yield chunk, header, "table" if is_table else "text"

        chunk_stream = unified_chunk_stream()

        # Process in full batches
        while True:
            batch_docs = []
            batch_metas = []
            batch_ids = []

            # Fill batch to batch_size from across sections
            try:
                for _ in range(batch_size):
                    chunk, section, chunk_type = next(chunk_stream)
                    if not chunk.strip():
                        continue
                    batch_docs.append(chunk)
                    batch_metas.append({
                        **metadata,
                        "chunk_number": chunk_id_counter,
                        "section": section,
                        "type": chunk_type
                    })
                    batch_ids.append(f"{collection_name}_chunk_{chunk_id_counter}")
                    chunk_id_counter += 1
            except StopIteration:
                if not batch_docs:
                    break

            # Add batch to ChromaDB
            try:
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                num_added = len(batch_docs)
                total_chunks += num_added
                logger.debug(f"Batch processed: {num_added} chunks (total: {total_chunks})")
            except Exception as e:
                logger.error(f"Failed to add batch to ChromaDB: {str(e)}")
                continue
            finally:
                del batch_docs, batch_metas, batch_ids
                gc.collect()

        logger.info(f"✅ Successfully processed and stored {total_chunks} chunks.")
        return total_chunks

    except Exception as e:
        logger.error(f"Error in process_document: {str(e)}", exc_info=True)
        raise
    finally:
        gc.collect()