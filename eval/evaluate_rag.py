import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# HF dataset (à ne pas confondre avec les classes Dataset internes à Ragas)
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Permet d'importer tes modules "rag.*" quand on lance via `python -m eval.evaluate_rag`
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# On réutilise tes embeddings LangChain côté Mistral
from rag.vector_pipe import MistralEmbeddings
from mistralai import Mistral
from langchain_core.language_models import LLM
from pydantic import Field, ConfigDict

# -----------------------------
# Config
# -----------------------------
load_dotenv(ROOT / ".env", override=True)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")
EVAL_FILE = ROOT / "eval" / "eval_data.json"

# -----------------------------
# Petit wrapper LLM (LangChain) pour Mistral
# (Ragas accepte directement un LLM LangChain et fera l’enveloppe,
# mais on garde ce wrapper minimal si tu veux ajuster le system prompt d’évaluation)
# -----------------------------
class MistralChatWrapper(LLM):
    client: object = Field(...)
    model: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "mistral-chat"

    def _call(self, prompt: str, stop=None):
        messages = [
            {"role": "system", "content": "Tu es un évaluateur. Réponds par une courte appréciation quand c’est requis."},
            {"role": "user", "content": prompt},
        ]
        resp = self.client.chat.complete(model=self.model, messages=messages)
        return resp.choices[0].message.content.strip()

# -----------------------------
# Charger le jeu d’éval
# -----------------------------
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

questions = [x["question"] for x in eval_data]
ground_truths = [x["ground_truth"] for x in eval_data]

# -----------------------------
# Interroger l’API pour obtenir answers + sources
# -----------------------------
answers = []
all_contexts = []   # liste de list[str]
has_any_context = False

for q in questions:
    try:
        r = requests.post(API_URL, json={"question": q}, timeout=60)
        r.raise_for_status()
        data = r.json()
        answers.append(data.get("answer", ""))

        # On tente de récupérer des "contexts" textuels depuis la réponse API
        # Cas 1: l’API renvoie des documents complets: {"page_content": "...", "metadata": {...}}
        raw_sources = data.get("sources", [])
        ctx_texts = []
        for s in raw_sources:
            if isinstance(s, dict):
                # plusieurs possibles: page_content direct, ou nested, ou rien
                if "page_content" in s and isinstance(s["page_content"], str):
                    ctx_texts.append(s["page_content"])
                elif "metadata" in s and isinstance(s["metadata"], dict) and "page_content" in s["metadata"]:
                    ctx_texts.append(str(s["metadata"]["page_content"]))
                # fallback léger: certaines API ne renvoient que titre+url -> pas utile pour les métriques contextuelles
        if ctx_texts:
            has_any_context = True
        all_contexts.append(ctx_texts)

    except Exception as e:
        print(f"⚠️ Erreur API pour `{q}` -> {e}")
        answers.append("")
        all_contexts.append([])

# -----------------------------
# Construire le HF Dataset comme attendu par Ragas
# Colonnes: question, ground_truth, answer, contexts
# -----------------------------
hf_ds = Dataset.from_dict({
    "question": questions,
    "ground_truth": ground_truths,
    "answer": answers,
    "contexts": all_contexts,  # liste de listes de strings
})

# -----------------------------
# LLM + Embeddings pour Ragas (via LangChain)
# (Ragas sait envelopper automatiquement les objets LangChain, cf. docs)
# -----------------------------
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
llm = MistralChatWrapper(client=mistral_client, model=os.getenv("EVAL_MODEL", "mistral-small-2503"))
embeddings = MistralEmbeddings()

# -----------------------------
# Choix des métriques selon présence de contexte
# - answer_relevancy: OK sans contexte (question + réponse)
# - faithfulness / context_precision: nécessitent du contexte
# -----------------------------
if has_any_context:
    metrics = [answer_relevancy, faithfulness, context_precision]
else:
    print("ℹ️ Aucun 'page_content' détecté dans les sources renvoyées par l’API ; "
          "évaluation limitée à 'answer_relevancy'. "
          "Pour des métriques plus riches, fais en sorte que l’API renvoie aussi le texte de contexte.")
    metrics = [answer_relevancy]

# -----------------------------
# Évaluation
# -----------------------------
results = evaluate(
    hf_ds,                  # << Hugging Face Dataset
    metrics=metrics,
    llm=llm,                # << LLM LangChain (Mistral) - Ragas l’enveloppe
    embeddings=embeddings   # << Embeddings LangChain (Mistral)
)

print("\n=== Résultats Ragas ===")
print(results)



