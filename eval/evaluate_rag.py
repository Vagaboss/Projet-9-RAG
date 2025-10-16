import os
import sys
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from ragas.run_config import RunConfig

from mistralai import Mistral
from langchain_core.language_models import LLM
from pydantic import Field, ConfigDict
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Config ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

load_dotenv(ROOT / ".env", override=True)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")
EVAL_FILE = ROOT / "eval" / "eval_data.json"
EVAL_MODEL = os.getenv("EVAL_MODEL", "mistral-small-2503")
EMB_MODEL = os.getenv("EVAL_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Wrapper LLM Mistral pour Ragas ---
class MistralChatWrapper(LLM):
    client: object = Field(...)
    model: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "mistral-chat"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs):
        messages = [
            {"role": "system", "content": "Tu es un évaluateur. Juge la qualité des réponses RAG."},
            {"role": "user", "content": prompt},
        ]
        resp = self.client.chat.complete(model=self.model, messages=messages)
        return resp.choices[0].message.content.strip()

# --- Charger le jeu de test ---
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_rows = json.load(f)

questions = [r["question"] for r in eval_rows]
ground_truths = [r["ground_truth"] for r in eval_rows]

answers = []
contexts = []
has_any_context = False

# --- Interroger l’API en respectant les limites ---
for q in questions:
    try:
        resp = requests.post(API_URL, json={"question": q}, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        answers.append(data.get("answer", ""))

        ctx_texts = []
        for s in data.get("sources", []):
            if isinstance(s, dict) and isinstance(s.get("page_content"), str):
                ctx_texts.append(s["page_content"])
        if ctx_texts:
            has_any_context = True
        contexts.append(ctx_texts)

    except Exception as e:
        print(f"⚠️ Erreur API pour `{q}` -> {e}")
        answers.append("")  # toujours remplir
        contexts.append([])

    # 🔑 respecter limite : 1 requête / seconde
    time.sleep(3)

# --- Construire Dataset HF ---
hf_ds = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
})

# --- LLM + embeddings ---
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
llm = MistralChatWrapper(client=mistral_client, model=EVAL_MODEL)

embeddings = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    encode_kwargs={"normalize_embeddings": True}
)

# --- Choix des métriques ---
if has_any_context:
    metrics = [answer_relevancy, faithfulness, context_precision, context_recall]
else:
    print("ℹ️ Aucun `page_content` détecté ; évaluation limitée à `answer_relevancy`.")
    metrics = [answer_relevancy]

# --- Lancer l’évaluation ---
run_cfg = RunConfig(max_workers=1, timeout=120)

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



