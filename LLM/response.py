from .setup import ask_claude
from .prompt import get_prompt

def generate_response(query: str, context: str, conversation_history: str = "") -> str:
    """
    Generate a response using Claude 3 Haiku via Aikipedia API
    
    Args:
        query: User's question
        context: Retrieved context from documents
        conversation_history: Previous conversation history (if any)
        
    Returns:
        str: Generated response from Claude 3 Haiku
    """
    try:
        # Format the prompt with context and conversation history
        prompt = get_prompt(context, conversation_history, query)
        
        # System prompt for Claude
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
        If you don't know the answer, say you don't know. Be concise and accurate in your responses."""
        
        # Get response from Claude 3 Haiku
        response = ask_claude(
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        return response.strip()
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"
