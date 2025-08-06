from typing import Tuple, List, Dict

def semantic_search(collection, query: str, n_results: int = 2) -> Dict:
    """
    Perform a semantic search on the provided ChromaDB collection.

    Args:
        collection: The ChromaDB collection object.
        query (str): The query string.
        n_results (int): Number of top results to retrieve.

    Returns:
        Dict: Dictionary containing documents, metadatas, distances, etc.
    """
    if not query.strip():
        raise ValueError("Query string cannot be empty.")
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results


def get_context_with_sources(results: Dict) -> Tuple[str, List[str]]:
    """
    Extract context and source information from search results.

    Args:
        results (Dict): The result from semantic_search().

    Returns:
        Tuple[str, List[str]]: Combined context string and list of sources.
    """
    if not results or 'documents' not in results or not results['documents'][0]:
        return "", []

    # Combine chunks from the first result
    context = "\n\n".join(results['documents'][0])

    sources = []
    for meta in results.get('metadatas', [[]])[0]:
        source = meta.get('source', 'Unknown source')
        chunk_id = meta.get('chunk', 'N/A')
        sources.append(f"{source} (chunk {chunk_id})")

    return context, sources
