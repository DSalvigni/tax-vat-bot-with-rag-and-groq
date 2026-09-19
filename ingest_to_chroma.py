import json
import os
import chromadb
import ollama
from tqdm import tqdm

JSON_PATH = "./temp/chunks_dataset.json"
CHROMA_PATH = "./chroma_db_iva"
MODEL_NAME = "bge-m3"
BATCH_SIZE = 15

class CustomOllamaEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = ollama.Client(timeout=300.0)

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        response = self.client.embed(model=self.model_name, input=input)
        return response.embeddings

def build_chroma_db():
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"File non trovato: {JSON_PATH}")

    print(f"1. Load of chunks from '{JSON_PATH}'...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   Loaded {len(chunks)} chunks.")

    print(f"2. Initializing embedding and ChromaDB (Model: {MODEL_NAME})...")
    embedding_fn = CustomOllamaEmbeddingFunction(model_name=MODEL_NAME)
    
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    collection = client.get_or_create_collection(
        name="iva_documents",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    print("3. Inserting or updating chunks and generating vectors...")
    
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Ingestion Progress"):
        batch = chunks[i : i + BATCH_SIZE]
        
        # Gestione ID: usa 'id' o 'chunk_id' se esistono, altrimenti genera chunk_{indice_globale}
        ids = []
        for idx, item in enumerate(batch):
            global_idx = i + idx
            chunk_id = item.get("id") or item.get("chunk_id") or f"chunk_{global_idx}"
            ids.append(str(chunk_id))

        documents = [item.get("text", item.get("content", "")) for item in batch]
        
        metadatas = [
            {
                "source": item.get("source", ""),
                "language": item.get("language", ""),
                "section": item.get("section", "")
            }
            for item in batch
        ]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    print(f"\nVector DB successfully created in '{CHROMA_PATH}'!")

if __name__ == "__main__":
    build_chroma_db()