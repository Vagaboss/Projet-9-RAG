"""
evaluate_ragas.py — Évaluation RAGAS utilisant Mistral comme LLM d'évaluation
"""

import sys
import os
import json
import time
import random
import logging
import warnings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_similarity, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_mistralai.chat_models import ChatMistralAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# --- Charger les modules du projet ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
from utils.vector_store import VectorStoreManager

# --- Configuration générale ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Initialisation du client Mistral et du Vector Store ---
client = MistralClient(api_key=MISTRAL_API_KEY)
vector_store_manager = VectorStoreManager()

# --- Prompt système ---
SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert de la NBA.
Tu réponds aux questions des analystes en t'appuyant sur les données contextuelles suivantes :

{context_str}

Question : {question}
Réponse :
"""

# --- Fonction principale pour obtenir réponse + contexte ---
def get_answer_and_context(question: str):
    try:
        logging.info(f"Recherche de contexte pour la question : {question}")
        search_results = vector_store_manager.search(question, k=SEARCH_K)

        context_str = "\n\n---\n\n".join([
            f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
            for res in search_results
        ]) if search_results else "Aucun contexte pertinent trouvé."

        final_prompt = SYSTEM_PROMPT.format(context_str=context_str, question=question)
        messages = [ChatMessage(role="user", content=final_prompt)]

        # Pause pour éviter le “Too Many Requests”
        time.sleep(random.uniform(2.5, 5.0))

        response = client.chat(model=MODEL_NAME, messages=messages, temperature=0.1)
        answer = response.choices[0].message.content if response.choices else "Réponse vide."

        return answer, context_str

    except Exception as e:
        logging.error(f"Erreur pendant la génération pour '{question}': {e}")
        return "", ""

# --- Charger le jeu d’évaluation ---
EVAL_FILE = os.path.join("eval", "eval_data.json")
if not os.path.exists(EVAL_FILE):
    raise FileNotFoundError("❌ Fichier eval_data.json introuvable dans le dossier 'eval/'")

with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

questions, answers, contexts, ground_truths = [], [], [], []

# --- Boucle principale ---
for i, item in enumerate(eval_data, 1):
    q = item["question"]
    gt = item["ground_truth"]

    logging.info(f"\n🧠 ({i}/{len(eval_data)}) Question : {q}")
    answer, ctx = get_answer_and_context(q)

    questions.append(q)
    answers.append(answer)
    contexts.append([ctx])  # ✅ Chaque contexte doit être une liste
    ground_truths.append(gt)

# --- Créer le Dataset compatible RAGAS ---
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

# --- Configurer Mistral comme LLM d’évaluation ---
llm_for_ragas = LangchainLLMWrapper(
    ChatMistralAI(api_key=MISTRAL_API_KEY, model=MODEL_NAME)
)

# --- Calcul des métriques RAGAS ---
logging.info("📊 Calcul des métriques RAGAS avec Mistral...")
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_similarity, context_precision, context_recall],
    llm=llm_for_ragas
)

# --- Afficher les résultats ---
print("\n===== 📈 RÉSULTATS RAGAS (Évaluation via Mistral) =====")
for metric, value in results.items():
    print(f"{metric}: {value:.3f}")

# --- Sauvegarder les résultats ---
RESULTS_PATH = os.path.join("eval", "results.json")
results_data = {
    "metrics": {k: float(v) for k, v in results.items()},
    "details": [
        {
            "question": q,
            "answer": a,
            "ground_truth": gt,
            "context": c[0]
        }
        for q, a, gt, c in zip(questions, answers, ground_truths, contexts)
    ]
}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=4, ensure_ascii=False)

logging.info(f"✅ Résultats enregistrés dans {RESULTS_PATH}")
