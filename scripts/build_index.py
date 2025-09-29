# scripts/build_index.py
import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rag.vector_pipe import rebuild_faiss

if __name__ == "__main__":
    print("🔄 Lancement de la reconstruction de l’index FAISS...")
    store_path = rebuild_faiss()
    print(f"✅ Index FAISS reconstruit et sauvegardé dans {store_path}")

