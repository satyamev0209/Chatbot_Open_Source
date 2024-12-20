# Import libraries
import yaml
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from utilities.fileOps import FileOps
from utilities.vectorizationOps import VectorizationOps
import utilities.llmOps 
from typing import List

# Initialize FastAPI app
app = FastAPI()

# Load configuration from YAML file
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Define request and response models
class Question(BaseModel):
    query: str

class Answer(BaseModel):
    answer: str

# Define the parameters for VectorizationOps
index_path = config['vector_ops']['index_path']
model_name = config['vector_ops']['model_name']

# Initialize vectorization operations with required parameters
vector_ops = VectorizationOps(index_path=index_path, model_name=model_name)

# Initialize FileOps with both base_dir and vector_ops
file_ops = FileOps(base_dir=config['file_ops']['base_dir'], vector_ops=vector_ops)

@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        chunks = await file_ops.save_file(file)
        return {"message": "File uploaded and processed", "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    try:
        if await file_ops.delete_file(filename):
            return {"message": "File and related embeddings deleted"}
        return {"message": "File not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
    
# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Q&A bot! Use /ask endpoint to ask questions."}

# Q&A endpoint
@app.post("/ask", response_model=Answer)
def ask_question(question: Question):
    try:
        context = vector_ops.search(question)
        answer = utilities.llmOps.generate_answer(context, question)
        return Answer(answer=answer)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating answer: {str(e)}"
        )

@app.post("/evaluate")
def evaluate(questions: List[str], expected_answers: List[str]):
    try:
        evaluation_results = evaluate_model(questions, expected_answers)
        return {"evaluation": evaluation_results}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during evaluation: {str(e)}"
        )
