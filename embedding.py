from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Globalna instancja modelu
EMBEDDING_MODEL = None
# Globalna zmienna przechowująca bazę i model
VECTOR_DATABASE = None


def get_embedding_model(model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"):
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = SentenceTransformer(model_name)
    return EMBEDDING_MODEL

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"):
        self.model = get_embedding_model(model_name)

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

class VectorDatabase:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.documents = [] 

    def add_documents(self, embeddings: np.ndarray, docs: list[dict]):
        print("add_documents")
        self.index.add(embeddings)
        self.documents.extend(docs)

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        print("searching")

        # Wyszukiwanie najbliższych wektorów
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            doc_info = self.documents[idx]
            results.append({
                "content": doc_info["content"],
                "source": doc_info["source"],
                "distance": float(dist)
            })
        return results


def build_vector_database():
    """
    Inicjalizuje bazę wektorową tylko raz i przechowuje ją globalnie.
    """
    global VECTOR_DATABASE
    if VECTOR_DATABASE is None:
        print("Initializing vector database for the first time...")
        # Ładowanie danych z pliku
        with open('data/processed/chunked_docs.json', 'r', encoding='utf-8') as f:
            chunked_docs = json.load(f)

        # Pobranie globalnego modelu
        embedding_model = EmbeddingModel()
        all_texts = [doc["content"] for doc in chunked_docs]
        embeddings = embedding_model.get_embeddings(all_texts)
        embedding_dim = embeddings.shape[1]

        # Tworzenie i zapisywanie bazy wektorowej
        db = VectorDatabase(embedding_dim=embedding_dim)
        db.add_documents(embeddings, chunked_docs)

        # Zapis do globalnej zmiennej
        VECTOR_DATABASE = (db, embedding_model)
    else:
        print("Using existing vector database.")
    return VECTOR_DATABASE

