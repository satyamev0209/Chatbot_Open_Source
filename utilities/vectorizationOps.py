import os
import json
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorizationOps:
    def __init__(self, index_path: str, model_name: str, metadata_path: str = "./metadata.json"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        self.embeddings = self.embedding_setup()
        self.index = None
        self.metadata = self._load_metadata()
        self.load_index()

    def embedding_setup(self):
        # Loading the embedding model from Hugging Face
        embedding_model_name = self.model_name
        model_kwargs = {"device": "cpu"}  # Change to "cpu" to use CPU
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs
        )
        return embeddings

    def load_index(self):
        try:
            if os.path.exists(self.index_path):
                self.index = FAISS.load_local(
                    self.index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
                # Create an empty FAISS index
                self.index = FAISS.from_texts(
                    texts=[""], 
                    embedding=self.embeddings
                )
                # Save the empty index
                self.index.save_local(self.index_path)
        except Exception as e:
            raise Exception(f"Error loading index: {str(e)}")

    def save_index(self, vectorstore):
        # faiss.write_index(self.index, self.index_path)
        # Persist the vectors locally on disk
        vectorstore.save_local(self.index_path)

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def process_file(self, file_path: str):
        print(f"\nProcessing file: {file_path}")
        
        # Load and split PDF into chunks
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=130)
        docs = text_splitter.split_documents(documents=documents)
        chunks = [doc.page_content for doc in docs]
        
        print(f"Number of chunks created: {len(chunks)}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Create or update the FAISS index
        if os.path.exists(self.index_path):
            # Load existing index
            existing_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
            # Add new documents to existing store
            existing_store.add_documents(docs)
            existing_store.save_local(self.index_path)
        else:
            # Create new vectorstore
            vectorstore = FAISS.from_documents(docs, self.embeddings)
            vectorstore.save_local(self.index_path)
        
        # Store metadata
        file_name = os.path.basename(file_path)
        self.metadata[file_name] = len(chunks)
        self._save_metadata()
        
        return len(chunks)

    def delete_embeddings(self, filename: str):
        if filename not in self.metadata:
            return False

        try:
            # Load current vectorstore
            current_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
            
            # Get all documents
            all_docs = current_store.similarity_search("", k=current_store.index.ntotal)
            
            # Filter out documents from the file to be deleted
            remaining_docs = [doc for doc in all_docs if filename not in doc.metadata.get('source', '')]
            
            # Create new vectorstore with remaining documents
            if remaining_docs:
                new_store = FAISS.from_documents(remaining_docs, self.embeddings)
                new_store.save_local(self.index_path)
            else:
                # If no documents remain, create empty index
                empty_store = FAISS.from_texts([""], self.embeddings)
                empty_store.save_local(self.index_path)
            
            # Update metadata
            del self.metadata[filename]
            self._save_metadata()
            
            return True
        except Exception as e:
            print(f"Error deleting embeddings: {str(e)}")
            return False

    def search(self, query: str) -> List[str]:
        print(f"\nSearching for query: {query}")
        persisted_vectorstore = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
        retriever = persisted_vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(query)
        print(f"Found {len(docs)} relevant documents")
        
        results = [doc.page_content for doc in docs]
        for i, result in enumerate(results):
            print(f"\nResult {i+1}: {result[:200]}...")  # Print first 200 chars of each result
        
        return results
