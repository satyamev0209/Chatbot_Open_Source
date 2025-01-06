from utilities.prompts import evaluation_prompt
from utilities.llmOps import generate_answer
from utilities.vectorizationOps import VectorizationOps
from langchain.llms import Ollama
from sklearn.metrics import accuracy_score, f1_score
from typing import List
import yaml

# Load config
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Initialize VectorizationOps with config parameters
vector_ops = VectorizationOps(
    index_path=config['vector_ops']['index_path'],
    model_name=config['vector_ops']['model_name']
)

def evaluate_model(questions: List[str], expected_answers: List[str]) -> dict:
    if len(questions) != len(expected_answers):
        raise ValueError("Number of questions and expected answers must match")
        
    llm = Ollama(model="llama3.1")
    generated_answers = []

    for question in questions:
        contexts = vector_ops.search(question)
        context = " ".join(contexts)
        answer = generate_answer(context, question)
        generated_answers.append(answer.strip())  # Clean up whitespace

    # Convert answers to binary (correct/incorrect) for scoring
    binary_expected = [1 if ans.strip() != "I Don't Know." else 0 for ans in expected_answers]
    binary_generated = [1 if ans.strip() != "I Don't Know." else 0 for ans in generated_answers]

    accuracy = accuracy_score(binary_expected, binary_generated)
    f1 = f1_score(binary_expected, binary_generated, average='weighted')

    return {
        "accuracy": float(accuracy),  # Convert numpy types to Python native types
        "f1_score": float(f1),
        "generated_answers": generated_answers,
        "num_questions": len(questions)
    }