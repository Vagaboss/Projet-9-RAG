import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from vector_pipe import MistralEmbeddings

# --- Charger .env ---
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

# --- Charger l’index FAISS sauvegardé ---
store_path = Path("data/faiss_store")
embeddings = MistralEmbeddings()
db = FAISS.load_local(str(store_path), embeddings, allow_dangerous_deserialization=True)

# --- Vérification 1 : cohérence index <-> métadonnées ---
index_size = db.index.ntotal
metadata_size = len(db.docstore._dict)
assert index_size == metadata_size, (
    f"Incohérence détectée : {index_size} vecteurs FAISS "
    f"mais {metadata_size} métadonnées"
)
print(f"✅ Vérification : {index_size} vecteurs et {metadata_size} métadonnées -> cohérents")

# --- Vérification 2 : nombre de chunks attendus ---
expected_count = 4728  # ajuster si tu régénères l’index
assert index_size == expected_count, (
    f"Erreur : attendu {expected_count} chunks, trouvé {index_size}"
)
print(f"✅ Vérification : {index_size} chunks bien indexés dans FAISS")

# --- Requête utilisateur ---
query = "concert de musique classique à Paris en avril 2025"
results = db.similarity_search(query, k=5)

print("\nRésultats de recherche :")
for r in results:
    print(f"- {r.metadata.get('title')} ({r.metadata.get('date_start')}) [{r.metadata.get('url')}]")
    print(f"  Extrait: {r.page_content[:200]}...\n")

