import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import os
from chromadb.config import Settings

class EmbeddingManager:
    def __init__(self, db_path="chroma_db", collection_name="code_embeddings"):
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(allow_reset=True)
        )
        
        self.embedding_function = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function # type: ignore
        )
        print(f"ChromaDB collection '{collection_name}' loaded/created with nomic-embed-text.")

    async def reset(self):
        print("--- Resetting ChromaDB ---")
        self.client.reset()
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_function # type: ignore
        )
        print("--- ChromaDB reset complete ---")

    async def index_file(self, file_path: str):
        if not os.path.exists(file_path):
            print(f"[WARN] File not found, cannot index: {file_path}")
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.collection.upsert(
                documents=[content],
                metadatas=[{"source": file_path}],
                ids=[file_path]
            )
            print(f"Indexed file: {file_path}")
        except Exception as e:
            print(f"Error indexing file {file_path}: {e}")

    async def retrieve_similar(self, query: str, n_results: int = 3) -> list:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results['documents'][0] if results and results['documents'] else []
        except Exception as e:
            print(f"Error retrieving similar documents: {e}")
            return []

    async def index_directory(self, directory: str):
        print(f"Starting to index directory: {directory}")
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.html', '.css', '.js')):
                    file_path = os.path.join(root, file)
                    await self.index_file(file_path)
        print("Directory indexing complete.") 