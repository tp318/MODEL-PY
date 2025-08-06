def get_prompt(context: str, conversation_history: str, query: str) -> str:
    """
    Generate a prompt by combining document context, conversation history, and the new user query.
    """
    prompt = f"""You are a helpful AI assistant that provides clear and concise answers based on the provided context.
Provide direct answers without using phrases like "according to the context" or "the document states".

Context from documents:
{context}

Previous conversation:
{conversation_history}

Please provide a response in the following format:


[short answer, 1–2 lines max]
Ref: [Clause number or section title if available]

If the answer cannot be determined from the context, respond with:

I cannot answer this based on the provided information.
Ref: N/A

Current question:
Q: {query}
"""

    return prompt
