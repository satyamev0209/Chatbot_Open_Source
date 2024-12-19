from typing import List
import requests
import prompts


def generate_answer(context: str, question: str):
    prompt = prompts.retrieval_prompt(context, question)
    # Initialize an instance of the Ollama model
    llm = Ollama(model="llama3.1")
    # Invoke the model to generate responses
    response = llm.invoke(prompt)
    return response