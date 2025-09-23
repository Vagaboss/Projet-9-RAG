import faiss
import json
import numpy as np
from mistralai import Mistral
import os

# Charger l’index FAISS
index = faiss.read_index("data/faiss_index.bin")

# Charger les métadonnées
with open("data/faiss_metadata.json", "r", encoding="utf-8") as f:
    metadatas = json.load(f)

print(f"Index chargé avec {index.ntotal} vecteurs")


# ✅ Vérification 1 : cohérence index <-> métadonnées
assert index.ntotal == len(metadatas), (
    f"Incohérence détectée : {index.ntotal} vecteurs FAISS "
    f"mais {len(metadatas)} métadonnées"
)
print("✅ Vérification : tous les vecteurs ont bien une métadonnée associée")

# ✅ Vérification 2 : l’index contient bien les 4728 chunks attendus
expected_count = 4728  # tu peux aussi mettre len(metadatas)
assert index.ntotal == expected_count, (
    f"Erreur : attendu {expected_count} chunks, trouvé {index.ntotal}"
)
print(f"✅ Vérification : {index.ntotal} chunks bien indexés dans FAISS")



# Requête utilisateur
query = "concert de musique classique à Paris en avril 2025"

# Obtenir embedding de la requête
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

response = client.embeddings.create(
    model="mistral-embed",
    inputs=[query]
)

query_vector = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)

# Recherche FAISS
k = 5  # nombre de résultats
distances, indices = index.search(query_vector, k)

print("\nRésultats de recherche :")
for idx, dist in zip(indices[0], distances[0]):
    meta = metadatas[idx]
    print(f"- {meta['title']} ({meta['date_start']}) [{meta['url']}]")
    print(f"  Distance: {dist:.4f}")
