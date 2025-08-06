import logging
from typing import Tuple, List, Dict, Any, Optional, Union
from .response import generate_response

logger = logging.getLogger(__name__)

def get_context_with_sources(search_results: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Extract context and sources from search results."""
    if not search_results:
        logger.warning("No search results provided to get_context_with_sources")
        return "", []
        
    context_parts = []
    sources = set()
    
    for i, result in enumerate(search_results, 1):
        try:
            if not isinstance(result, dict):
                logger.warning(f"Unexpected result type at index {i}: {type(result)}")
                continue
                
            document = result.get('document')
            metadata = result.get('metadata', {})
            
            if not document or not isinstance(document, str):
                logger.warning(f"Missing or invalid document at index {i}")
                continue
                
            context_parts.append(f"Context {i}:\n{document}")
            
            source = metadata.get('source')
            if source and isinstance(source, str):
                sources.add(source)
                
        except Exception as e:
            logger.error(f"Error processing result {i}: {str(e)}", exc_info=True)
    
    return "\n\n".join(context_parts), list(sources)


def rag_query(
    collection, 
    query: Union[str, List[str]], 
    n_chunks: int = 3, 
    conversation_history: str = ""
) -> Union[Tuple[str, List[str]], List[Tuple[str, List[str]]]]:
    """
    Perform Retrieval-Augmented Generation query.
    Can handle single query or list of queries (batch mode).
    Now sends only ONE LLM call for multiple questions, while preserving per-question context.
    
    Args:
        collection: ChromaDB collection for semantic search
        query: Single question or list of questions
        n_chunks: Number of chunks to retrieve per query
        conversation_history: Previous conversation history (if any)
        
    Returns:
        For single query: (answer, sources)
        For multiple queries: [(answer, sources), ...]
    """
    logger.info(f"Starting RAG query with input of type: {type(query)}")

    if not collection:
        error_msg = "No collection provided for RAG query"
        logger.error(error_msg)
        if isinstance(query, list):
            return [(error_msg, []) for _ in query]
        return error_msg, []

    if not query:
        error_msg = "Invalid query provided"
        logger.error(error_msg)
        if isinstance(query, list):
            return [(error_msg, []) for _ in query]
        return error_msg, []

    # Handle single query by converting to list
    is_single_query = isinstance(query, str)
    queries = [query] if is_single_query else query

    try:
        # 1. Perform batch semantic search
        logger.info(f"Performing batch semantic search for {len(queries)} queries")
        try:
            results = collection.query(
                query_texts=queries,
                n_results=n_chunks
            )
            logger.debug(f"Raw batch search results received")
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}", exc_info=True)
            if is_single_query:
                return f"Error searching documents: {str(e)}", []
            return [(f"Error searching documents: {str(e)}", []) for _ in queries]

        # 2. Build per-question context and collect sources
        contexts = []
        all_sources = set()

        for idx, query_text in enumerate(queries):
            if not results or 'documents' not in results or len(results['documents']) <= idx:
                contexts.append("")
                continue

            documents = results['documents'][idx] or []
            metadatas = results['metadatas'][idx] or []

            search_results = []
            for doc, meta in zip(documents, metadatas):
                if isinstance(meta, dict):
                    source = meta.get('source')
                    if source and isinstance(source, str):
                        all_sources.add(source)
                search_results.append({
                    'document': str(doc) if doc is not None else "",
                    'metadata': meta if isinstance(meta, dict) else {}
                })

            context, _ = get_context_with_sources(search_results)
            contexts.append(context)

        sources_list = list(all_sources)

        # 3. Combine all questions and contexts into one prompt
        prompt_parts = ["You are an expert insurance policy assistant. Answer each question using only the provided context."]
        if conversation_history.strip():
            prompt_parts.append(f"Conversation History:\n{conversation_history}")

        for i, (q, ctx) in enumerate(zip(queries, contexts)):
            prompt_parts.append(f"\nQuestion {i+1}: {q}")
            if ctx.strip():
                prompt_parts.append(f"Context:\n{ctx}")
            else:
                prompt_parts.append("Context: No relevant information found.")

        prompt_parts.append("\nProvide answers in this format:")
        for i in range(1, len(queries) + 1):
            prompt_parts.append(f"{i}. [Answer]")
        
        prompt = "\n".join(prompt_parts)

        # 4. Generate response with a single LLM call
        try:
            logger.info(f"Generating response for {len(queries)} questions in a single LLM call...")
            final_response = generate_response(prompt, context="", conversation_history="")
            logger.info("Received response from LLM")
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            error_answer = f"Error generating response: {str(e)}"
            if is_single_query:
                return error_answer, sources_list
            return [(error_answer, sources_list) for _ in queries]

        # 5. Parse answers (split by newlines and remove numbering)
        import re
        # Split by newline and remove any leading numbers/dots/whitespace
        raw_answers = [line.strip() for line in final_response.strip().split('\n') if line.strip()]
        # Remove any remaining numbering patterns (e.g., "1. " or "1) ")
        parsed_answers = [re.sub(r'^\s*\d+[.)]?\s*', '', ans) for ans in raw_answers]

        # Pad if needed
        while len(parsed_answers) < len(queries):
            parsed_answers.append("Answer not available in the policy document.")

        # 6. Format output
        answers = [(ans, sources_list) for ans in parsed_answers]

        # 7. Return appropriately
        if is_single_query:
            logger.info("Returning single answer")
            return answers[0]
        else:
            logger.info(f"Batch processing complete. Generated {len(answers)} answers in one LLM call.")
            return answers

    except Exception as e:
        logger.critical(f"Unexpected error in rag_query: {str(e)}", exc_info=True)
        if is_single_query:
            return f"An unexpected error occurred: {str(e)}", []
        return [(f"An unexpected error occurred: {str(e)}", []) for _ in queries]