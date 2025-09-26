import os
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
import time

from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()

# --- Fonction pour extraire les métadonnées ---
def metadata_extractor(record, metadata):
    return {
        "id": record.get("id"),
        "title": record.get("title"),
        "url": record.get("url"),
        "date_start": record.get("date_start"),
        "date_end": record.get("date_end"),
        "city": record.get("city"),
        "region": record.get("region"),
        "keywords": record.get("keywords"),
    }

# --- Wrapper embeddings Mistral ---
class MistralEmbeddings(Embeddings):
    def __init__(self, model="mistral-embed"):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("⚠️ La clé API Mistral n'est pas définie.")
        self.client = Mistral(api_key=api_key)
        self.model = model

    def embed_documents(self, texts):
        """Embeddings pour une liste de documents"""
        embeddings = []
        batch_size = 50
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                inputs=batch
            )
            embeddings.extend([e.embedding for e in response.data])
            time.sleep(1)  # respecter limite 1 req/sec
        return embeddings

    def embed_query(self, text):
        """Embedding pour une seule requête"""
        response = self.client.embeddings.create(
            model=self.model,
            inputs=[text]
        )
        return response.data[0].embedding


if __name__ == "__main__":
    # --- Charger les données JSON ---
    loader = JSONLoader(
        file_path="./data/events_clean.json",
        jq_schema=".[]",
        content_key="text_to_embed",
        metadata_func=metadata_extractor
    )

    docs = loader.load()
    print(f"Documents chargés : {len(docs)}")
    pprint(docs[0].metadata)

    # --- Split en chunks ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(docs)
    print(f"Nombre de chunks générés : {len(split_docs)}")

    # --- Créer l’index FAISS avec LangChain ---
    embeddings = MistralEmbeddings()
    db = FAISS.from_documents(split_docs, embeddings)

    # --- Sauvegarder l’index ---
    store_path = Path("data/faiss_store")
    db.save_local(str(store_path))
    print(f"💾 Index FAISS + métadonnées sauvegardé dans {store_path}")


# fonction rebuild pour API
def rebuild_faiss():
    """
    Reconstruit l’index FAISS à partir du fichier events_clean.json
    et le sauvegarde dans data/faiss_store.
    """
    print("🔄 Reconstruction de l’index FAISS en cours...")

    # --- Charger les données JSON ---
    loader = JSONLoader(
        file_path="./data/events_clean.json",
        jq_schema=".[]",
        content_key="text_to_embed",
        metadata_func=metadata_extractor
    )
    docs = loader.load()
    print(f"Documents chargés : {len(docs)}")

    # --- Split en chunks ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(docs)
    print(f"Nombre de chunks générés : {len(split_docs)}")

    # --- Créer l’index FAISS avec LangChain ---
    embeddings = MistralEmbeddings()
    db = FAISS.from_documents(split_docs, embeddings)

    # --- Sauvegarder l’index ---
    store_path = Path("data/faiss_store")
    db.save_local(str(store_path))

    print(f"✅ Index FAISS reconstruit et sauvegardé dans {store_path}")
    return store_path

