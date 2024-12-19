# Import libraries
import yaml
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from utilities.fileOps import FileOps
from utilities.vectorizationOps import VectorizationOps
import utilities.llmOps 

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

# Initialize components
file_ops = FileOps(base_dir=config['file_ops']['base_dir'])
vector_ops = VectorizationOps(
    index_path=config['vector_ops']['index_path'],
    model_name=config['vector_ops']['model_name']
)

@app.post("/upload")
def upload_file(file: UploadFile):
    try:
        chunks = file_ops.save_file(file)
        return {"message": "File uploaded and processed", "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.delete("/delete/{filename}")
def delete_file(filename: str):
    try:
        if file_ops.delete_file(filename):
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
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")






