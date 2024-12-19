import os
from fastapi import UploadFile
from typing import List
from vectorizationOps import VectorizationOps

class FileOps:
    def __init__(self, base_dir: str, vector_ops: VectorizationOps):
        self.base_dir = base_dir
        self.vector_ops = vector_ops
        os.makedirs(self.base_dir, exist_ok=True)

    def save_file(self, file: UploadFile):
        file_path = os.path.join(self.base_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        chunks = self.vector_ops.process_file(file_path)
        return chunks

    def delete_file(self, filename: str):
        file_path = os.path.join(self.base_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            # Delete embeddings associated with the file
            embeddings_deleted = self.vector_ops.delete_embeddings(filename)
            return embeddings_deleted
        return False

    def list_files(self):
        return [f for f in os.listdir(self.base_dir) if os.path.isfile(os.path.join(self.base_dir, f))]

