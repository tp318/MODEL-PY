def add_to_collection(collection, ids, texts, metadatas):
    """
    Add documents to the ChromaDB collection in batches of 100.
    """
    if not texts:
        print("No texts to add.")
        return

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        collection.add(
            documents=texts[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )


def process_and_add_documents(collection, folder_path: str):
    """
    Process all supported documents in a folder and add them to the ChromaDB collection.
    """
    supported_extensions = {".pdf", ".docx", ".txt"}

    files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file))
        and os.path.splitext(file)[1].lower() in supported_extensions
    ]

    for file_path in files:
        print(f"\n📄 Processing: {os.path.basename(file_path)}")
        ids, texts, metadatas = process_document(file_path)

        if texts:
            add_to_collection(collection, ids, texts, metadatas)
            print(f"✅ Added {len(texts)} chunks to collection.")
        else:
            print(f"⚠️ Skipped {os.path.basename(file_path)} — no readable content.")
