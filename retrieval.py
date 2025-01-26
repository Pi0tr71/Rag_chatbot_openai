import os
from typing import List, Dict
from embedding import build_vector_database

def retrieve_docs(
    user_query: str,
    top_k: int = 3
) -> List[Dict]:
    """
    Funkcja pobierająca najbardziej pasujące fragmenty tekstu z bazy wektorowej
    na podstawie zapytania użytkownika.
    
    Zwraca listę słowników:
    [
      {
        "content": "<treść chunku>",
        "source": "nazwa_pliku.txt",
        "distance": <wartość_odległości>
      },
      ...
    ]
    """
    print("retrieve_docs")
    db, embedding_model = build_vector_database()
    query_embedding = embedding_model.get_embeddings([user_query])
    results = db.search(query_embedding, top_k=top_k)
    
    return results

def build_prompt(
    system_prompt: str,
    context_chunks: List[Dict],
    user_question: str
) -> str:
    """
    Funkcja budująca finalny prompt dla modelu LLM:
      - systemowy prompt (z instrukcjami),
      - znalezione fragmenty (kontekst),
      - pytanie użytkownika.
    """
    print("build_prompt")
    context_text = "\n".join(
        [
            f"{idx+1}. {chunk['content']} (Source: {chunk['source']})"
            for idx, chunk in enumerate(context_chunks)
        ]
    )

    final_prompt = (
        f"{system_prompt}\n\n"
        f"Oto fragmenty kontekstu:\n"
        f"{context_text}\n\n"
        f"Pytanie: {user_question}\n"
        f"Odpowiedź:"
    )
    
    return final_prompt