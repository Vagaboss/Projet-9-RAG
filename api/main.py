# main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from rag.chatbot import answer_question   # ta fonction qui interroge le RAG
from rag.vector_pipe import rebuild_faiss # ta fonction qui reconstruit FAISS

# --- Initialiser FastAPI ---
app = FastAPI(
    title="RAG Chatbot API",
    description="API REST pour poser des questions sur les événements de Paris (OpenAgenda) via un système RAG.",
    version="1.0.0",
)

# --- Modèle d'entrée pour /ask ---
class AskRequest(BaseModel):
    question: str




# --- Endpoint health ---
@app.get("/health")
def health():
    """Vérifie que l’API fonctionne."""
    return {"status": "ok", "message": "API RAG opérationnelle"}

# --- Endpoint /ask ---
@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")

    print(f"📩 Question reçue : {req.question}")

    try:
        # Utilise ta fonction RAG
        answer, sources = answer_question(req.question)
        print("✅ Réponse générée avec succès")
    except Exception as e:
        import traceback
        print("❌ Erreur dans answer_question :", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "answer": answer,
        "sources": [
            {
                "title": d.metadata.get("title"),
                "url": d.metadata.get("url"),
                "date_start": d.metadata.get("date_start"),
                "date_end": d.metadata.get("date_end"),
                "city": d.metadata.get("city"),
                "region": d.metadata.get("region"),
                "keywords": d.metadata.get("keywords"),
                "page_content": d.page_content
            }
            for d in sources
        ]
    }

# --- Endpoint /rebuild ---
@app.post("/rebuild")
def rebuild():
    """Reconstruit l’index FAISS à partir des données JSON (events_clean.json)."""
    store_path = rebuild_faiss()
    return {"status": f"Index reconstruit et sauvegardé dans {store_path}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True, log_level="debug")
