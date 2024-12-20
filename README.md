# Retrieval-Augmented Generation (RAG) Pipeline

This repository implements an end-to-end RAG pipeline with file uploads, FAISS vector indexing, and LLM-based question answering.

## Features
- File upload and deletion
- Automatic document chunking and FAISS indexing
- Retrieval-augmented generation with an LLM (e.g., LLama)

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/satyamev0209/Chatbot_Open_Source
   ```
2. Navigate to the project directory:
   ```bash
   cd Chatbot_Open_Source
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start Ollama server:
   ```bash
   ollama start
   ```
6. Pull the required model:
   ```bash
   ollama pull <model-name>
   ```
7. Update `config.yaml` with your settings.

## Usage
1. Start the application:
   ```bash
   python app.py
   ```
2. Open your web browser and go to `http://localhost:5000` to access the application.

## API Endpoints
- `POST /upload`: Upload document files
- `DELETE /delete/{filename}`: Remove documents
- `POST /ask`: Ask questions about documents
- `POST /evaluate`: Evaluate model performance

## API Examples
- **Ask a question:**
   ```bash
   curl -X POST "http://localhost:5000/ask" -H "Content-Type: application/json" -d '{"question": "What is RAG?"}'
   ```
- **Upload a document:**
   ```bash
   curl -X POST "http://localhost:5000/upload" -F "file=@/path/to/your/document.pdf"
   ```

## Performance Evaluation
Use the `/evaluate` endpoint to assess model performance with custom test sets.

## Project Structure
```
Chatbot_Open_Source/
├── app.py
├── config.yaml
├── requirements.txt
├── ...
```

## License
MIT
