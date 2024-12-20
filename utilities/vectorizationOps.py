import os
import json
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


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
        if os.path.exists(self.index_path):
            self.index = FAISS.load_local(
                self.index_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            # Create an empty FAISS index
            self.index = FAISS.from_texts(
                texts=[""], 
                embedding=self.embeddings
            )

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
        # Load and split PDF into chunks
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=30, separator="\n")
        docs = text_splitter.split_documents(documents=documents)
        chunks = [doc.page_content for doc in docs]

        # Create or update the FAISS index
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        
        # Store the current document in metadata
        file_name = os.path.basename(file_path)
        self.metadata[file_name] = len(chunks)  # Store number of chunks instead of indices
        self._save_metadata()
        
        # Save the updated index
        vectorstore.save_local(self.index_path)
        
        return len(chunks)

    def delete_embeddings(self, filename: str):
        if filename not in self.metadata:
            return False

        # Retrieve the indices associated with the file
        indices_to_delete = self.metadata[filename]
        indices_to_delete.sort()  # Ensure indices are sorted

        # Mark embeddings as "deleted" in a new FAISS index
        all_embeddings = self.index.reconstruct_n(0, self.index.ntotal)
        remaining_embeddings = [
            all_embeddings[i] for i in range(len(all_embeddings)) if i not in indices_to_delete
        ]

        # Rebuild the FAISS index
        self.index = FAISS.IndexFlatL2(768)
        self.index.add(remaining_embeddings)
        self.save_index()

        # Update metadata
        del self.metadata[filename]
        self._save_metadata()
        return True

    def search(self, query: str) -> List[str]:
        # query_embedding = self.model.encode([query])
        # distances, indices = self.index.search(query_embedding, top_k)
        # return indices.flatten().tolist()
        # Load from local storage
        persisted_vectorstore = FAISS.load_local(self.index_path, self.embeddings,allow_dangerous_deserialization=True)
        #creating a retriever on top of database
        retriever = persisted_vectorstore.as_retriever()
