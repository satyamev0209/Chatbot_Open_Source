from utilities.prompts import LLMEvaluationTemplate
from llmOps import generate_answer
from langchain.llms import Ollama
from sklearn.metrics import accuracy_score, f1_score
from typing import List

def evaluate_model(questions: List[str], expected_answers: List[str], vector_ops) -> dict:
    llm = Ollama(model="llama3.1")  # Initialize the model
    generated_answers = []

    for question in questions:
        context = vector_ops.search(question)  # Retrieve context for the question
        answer = generate_answer(context, question)  # Generate answer
        generated_answers.append(answer)

    accuracy = accuracy_score(expected_answers, generated_answers)
    f1 = f1_score(expected_answers, generated_answers, average='weighted')

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "generated_answers": generated_answers
    }