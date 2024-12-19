import os
import json
from pypdf_loader import PyPDFLoader
import faiss
from typing import List
from sentence_transformers import SentenceTransformer


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
        #loading the embedding model from huggingface
        embedding_model_name = self.model_name
        model_kwargs = {"device": "cuda"}
        embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs
        )
        return embeddings

    def load_index(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(768)  # Assuming 768-dimensional embeddings

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
        documents = loader.load()        #Splitting the data into chunk
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        docs = text_splitter.split_documents(documents=documents)

        # # Generate embeddings
        # embeddings = self.model.encode(chunks)
        #loading the data and correspond embedding into the FAISS
        vectorstore = FAISS.from_documents(docs, self.embeddings)

        # Add embeddings to FAISS index
        start_index = self.index.ntotal
        self.index.add(self.embeddings)
        self.save_index(vectorstore)

        # Update metadata
        file_name = os.path.basename(file_path)
        self.metadata[file_name] = list(range(start_index, start_index + len(chunks)))
        self._save_metadata()

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
        self.index = faiss.IndexFlatL2(768)
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
