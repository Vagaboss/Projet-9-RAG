# eval/evaluate_rag.py
import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

from datasets import Dataset

# Ragas
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from ragas.run_config import RunConfig

# LLM d'évaluation (Mistral) via LangChain
from mistralai import Mistral
from langchain_core.language_models import LLM
from pydantic import Field, ConfigDict

# Embeddings pour l'évaluation Ragas (locaux et stables)
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- rendre importable le package local quand on lance via `python -m eval.evaluate_rag`
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

load_dotenv(ROOT / ".env", override=True)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")
EVAL_FILE = ROOT / "eval" / "eval_data.json"
EVAL_MODEL = os.getenv("EVAL_MODEL", "mistral-small-2503")  # modèle Mistral pour l'évaluation
EMB_MODEL = os.getenv("EVAL_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # rapide/robuste

# -----------------------------
# Wrapper LLM (LangChain) compatible Ragas
# -----------------------------
class MistralChatWrapper(LLM):
    client: object = Field(...)
    model: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "mistral-chat"

    # IMPORTANT: accepter **kwargs pour compatibilité LangChain/Ragas
    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs):
        messages = [
            {"role": "system", "content": "Tu es un évaluateur. Juge la qualité des réponses RAG."},
            {"role": "user", "content": prompt},
        ]
        resp = self.client.chat.complete(model=self.model, messages=messages)
        return resp.choices[0].message.content.strip()

# -----------------------------
# 1) Charger le jeu de test (questions + ground_truth)
# -----------------------------
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_rows = json.load(f)

questions = [r["question"] for r in eval_rows]
ground_truths = [r["ground_truth"] for r in eval_rows]

# -----------------------------
# 2) Interroger l’API /ask et récupérer answer + sources (avec page_content)
# -----------------------------
answers = []
contexts = []            # liste de listes de strings (texte)
has_any_context = False

for q in questions:
    try:
        resp = requests.post(API_URL, json={"question": q}, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # réponse générée par ton RAG
        answers.append(data.get("answer", ""))

        # construire un contexte textuel (liste[str]) à partir de `sources`
        ctx_texts = []
        for s in data.get("sources", []):
            # cas idéal: chaque source contient page_content (texte du chunk)
            if isinstance(s, dict) and isinstance(s.get("page_content"), str):
                ctx_texts.append(s["page_content"])
        if ctx_texts:
            has_any_context = True
        contexts.append(ctx_texts)

    except Exception as e:
        print(f"⚠️ Erreur API pour `{q}` -> {e}")
        answers.append("")
        contexts.append([])

# -----------------------------
# 3) Construire le Dataset HuggingFace attendu par Ragas
#    colonnes: question, answer, contexts, ground_truth
# -----------------------------
hf_ds = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,       # liste de listes (peut être vide)
    "ground_truth": ground_truths,
})

# -----------------------------
# 4) Préparer LLM & embeddings pour Ragas
# -----------------------------
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
llm = MistralChatWrapper(client=mistral_client, model=EVAL_MODEL)

embeddings = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    encode_kwargs={"normalize_embeddings": True}
)

# -----------------------------
# 5) Choix des métriques
# -----------------------------
if has_any_context:
    metrics = [answer_relevancy, faithfulness, context_precision, context_recall]
else:
    print("ℹ️ Aucun `page_content` détecté dans les sources de l’API ; "
          "évaluation limitée à `answer_relevancy`. "
          "Ajoute le texte des chunks dans /ask pour activer fidelity/precision/recall.")
    metrics = [answer_relevancy]

# -----------------------------
# 6) Lancer l’évaluation Ragas (limiter la concurrence = important pour Mistral)
# -----------------------------
run_cfg = RunConfig(max_workers=1)  # 1 req/s -> éviter 429/timeout

results = evaluate(
    hf_ds,
    metrics=metrics,
    llm=llm,
    embeddings=embeddings,
    run_config=run_cfg,
    show_progress=True,
    raise_exceptions=False,
)

print("\n=== Résultats Ragas ===")
print(results)

