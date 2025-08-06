import os

def process_document(file_path: str):
    """
    Process a single document:
    - Read and extract the text,
    - Split into sentence-based chunks,
    - Create metadata and chunk IDs.
    Returns: (ids, chunks, metadatas)
    """
    try:
        # Read the document using correct reader
        content = read_document(file_path)

        # Split content into chunks
        chunks = split_text(content)

        # Create metadata and IDs
        file_name = os.path.basename(file_path)
        metadatas = [{"source": file_name, "chunk": i} for i in range(len(chunks))]
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]

        return ids, chunks, metadatas

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], [], []
