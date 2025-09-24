import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# Importer ta fonction chatbot
from chatbot import answer_question  

# --- Charger variables d'environnement ---
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

# --- Charger modèle de similarité ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Jeu de test (10 cas) ---
test_cases = [
    {
        "question": "Quels concerts de musique classique en avril 2025 à Paris ?",
        "expected": "Concert de l'Ensemble Marani le 6 avril 2025 à Paris",
    },
    {
        "question": "Y a-t-il un hommage à Beethoven en 2025 ?",
        "expected": "Hommage à Beethoven en avril 2025 à Paris",
    },
    {
        "question": "Quels festivals ont lieu à Paris en juin 2025 ?",
        "expected": "Festival de la musique en juin 2025 à Paris",
    },
    {
        "question": "Des événements culinaires sont-ils prévus en juin 2025 ?",
        "expected": "Atelier ou festival culinaire en juin 2025 à Paris",
    },
    {
        "question": "Je cherche une exposition d’art en mai 2025 à Paris",
        "expected": "Exposition d'art en mai 2025 à Paris",
    },
    {
        "question": "Quels événements pour les familles en mars 2025 ?",
        "expected": "Événements famille en mars 2025 à Paris",
    },
    {
        "question": "Y a-t-il des événements gratuits en avril 2025 ?",
        "expected": "Événements gratuits à Paris en avril 2025",
    },
    {
        "question": "Un spectacle de théâtre est-il prévu en février 2025 à Paris ?",
        "expected": "Spectacle de théâtre en février 2025 à Paris",
    },
    {
        "question": "Quels événements sportifs sont programmés en 2025 à Paris ?",
        "expected": "Événements sportifs en 2025 à Paris",
    },
    {
        "question": "Un concert jazz est-il prévu en mai 2025 à Paris ?",
        "expected": "Concert de jazz en mai 2025 à Paris",
    },
]

# --- Évaluation ---
scores = []

for case in test_cases:
    q = case["question"]
    expected = case["expected"]

    print(f"\n🔹 Question: {q}")
    predicted, _ = answer_question(q)  # chatbot génère la réponse

    # Similarité
    emb_expected = model.encode(expected, convert_to_tensor=True)
    emb_predicted = model.encode(predicted, convert_to_tensor=True)
    score = util.cos_sim(emb_expected, emb_predicted).item()

    scores.append(score)

    print(f"✅ Expected: {expected}")
    print(f"🤖 Predicted: {predicted}")
    print(f"📊 Similarité: {score:.3f}")

# --- Résumé ---
avg_score = sum(scores) / len(scores)
print("\n=== Résultats globaux ===")
print(f"Score moyen de similarité: {avg_score:.3f}")
