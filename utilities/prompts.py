def retrieval_prompt(context: str, question: str) -> str:
    """
    Generates a prompt for retrieval-augmented generation.
    """
    return f"""
    Generate answer of given Question strictly after reading the given Context. If question is out of context then simply give 'I Don't Know.' nothing else. 
    
    ---

    Context:
    {context}
    
    ---
    
    Question:
    {question}
    
    """

def evaluation_prompt(question: str, expected_answer: str, generated_answer: str) -> str:
    return f"""
    You are a LLM tester bot.
    Evaluate the given generated answer for the question.
    """