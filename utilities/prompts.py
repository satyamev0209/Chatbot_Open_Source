def retrieval_prompt(context: str, question: str) -> str:
    """
    Generates a prompt for retrieval-augmented generation.
    """
    return f"""
    You are a QNA bot.
    Generate answer of given Question strictly from given Context. If question is out of context then simply give 'I Don't Know.' nothing else. 
    
    ---

    Context:
    {context}
    
    ---
    
    Question:
    {question}
    
    """
