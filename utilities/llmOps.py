from langchain_community.llms import Ollama
from utilities import prompts


def generate_answer(context: str, question: str):
    prompt = prompts.retrieval_prompt(context, question)
    llm = Ollama(model="llama3.1")
    return llm.invoke(prompt)