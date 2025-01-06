from langchain_ollama.chat_models import ChatOllama
from utilities import prompts


def generate_answer(context: str, question: str):
    prompt = prompts.retrieval_prompt(context, question)
    llm = ChatOllama(model="llama3.2:1b")
    response = llm.invoke(prompt)
    # Convert response to string if it's not already
    return str(response.content) if hasattr(response, 'content') else str(response)