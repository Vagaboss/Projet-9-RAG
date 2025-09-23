import os
import json
import time
import numpy as np
import faiss
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mistralai import Mistral

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
    print("Exemple chunk :", split_docs[0].page_content[:200], "...")
    print("Exemple metadata :", split_docs[0].metadata)

    # --- Préparer les textes et métadonnées ---
    texts = [d.page_content for d in split_docs]
    metadatas = [d.metadata for d in split_docs]

    # --- Initialiser le client Mistral ---
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("⚠️ La clé API Mistral n'est pas définie.")

    client = Mistral(api_key=api_key)

    # --- Vectorisation avec batching ---
    print("Vectorisation avec Mistral en cours...")
    batch_size = 50
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model="mistral-embed",
            inputs=batch
        )
        all_embeddings.extend([e.embedding for e in response.data])
        time.sleep(1)  # respecter la limite 1 req/sec

    vectors = np.array(all_embeddings, dtype="float32")
    print(f"✅ {vectors.shape[0]} embeddings générés")

    # --- Création de l’index FAISS ---
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    print(f"✅ {index.ntotal} chunks indexés dans FAISS")

    # --- Sauvegarde ---
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    faiss_index_path = data_dir / "faiss_index.bin"
    faiss.write_index(index, str(faiss_index_path))

    metadata_path = data_dir / "faiss_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

    print(f"💾 Index FAISS sauvegardé dans {faiss_index_path}")
    print(f"💾 Métadonnées sauvegardées dans {metadata_path}")


