# main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel

from rag.chatbot import answer_question   # ta fonction qui interroge le RAG
from rag.vector_pipe import rebuild_faiss # ta fonction qui reconstruit FAISS

# --- Initialiser FastAPI ---
app = FastAPI(
    title="RAG Chatbot API",
    description="API REST pour poser des questions sur les événements de Paris (OpenAgenda) via un système RAG.",
    version="1.0.0",
)

# --- Modèle d'entrée pour /ask ---
class QuestionPayload(BaseModel):
    question: str

# --- Endpoint health ---
@app.get("/health")
def health():
    """
    Vérifie que l’API fonctionne.
    """
    return {"status": "ok", "message": "API RAG opérationnelle"}

# --- Endpoint /ask ---
@app.post("/ask")
def ask_question(payload: QuestionPayload):
    """
    Pose une question au chatbot RAG et retourne une réponse augmentée + sources.
    """
    if not payload.question.strip():
        return {"error": "Veuillez fournir une question valide."}

    answer, sources = answer_question(payload.question)
    return {
        "question": payload.question,
        "answer": answer,
        "sources": [s.metadata for s in sources]
    }

# --- Endpoint /rebuild ---
@app.post("/rebuild")
def rebuild():
    """
    Reconstruit l’index FAISS à partir des données JSON (events_clean.json).
    """
    store_path = rebuild_faiss()
    return {"status": f"Index reconstruit et sauvegardé dans {store_path}"}
