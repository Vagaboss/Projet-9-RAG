# eval/evaluate_rag.py
import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
import time

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision
from ragas.run_config import RunConfig

# Rendez importable "rag.*" même en lançant `python -m eval.evaluate_rag`
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rag.vector_pipe import MistralEmbeddings  # tes embeddings Mistral
from mistralai import Mistral
from langchain_core.language_models import LLM
from pydantic import Field, ConfigDict

# ---------- Config ----------
load_dotenv(ROOT / ".env", override=True)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")
EVAL_FILE = ROOT / "eval" / "eval_data.json"
EVAL_MODEL = os.getenv("EVAL_MODEL", "mistral-small-2503")

# ---------- Wrapper LLM compatible LangChain ----------
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
            {"role": "system", "content": "Tu es un évaluateur. Réponds brièvement quand requis."},
            {"role": "user", "content": prompt},
        ]
        resp = self.client.chat.complete(model=self.model, messages=messages)
        return resp.choices[0].message.content.strip()

# ---------- Charger jeu d'éval ----------
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

questions = [x["question"] for x in eval_data]
ground_truths = [x["ground_truth"] for x in eval_data]

# ---------- Interroger l’API ----------
answers = []
contexts = []      # List[List[str]]
has_any_context = False

for q in questions:
    try:
        r = requests.post(API_URL, json={"question": q}, timeout=60)
        r.raise_for_status()
        data = r.json()

        answers.append(data.get("answer", ""))

        # On récupère un éventuel texte de contexte (page_content)
        raw_sources = data.get("sources", [])
        ctx_texts = []
        for s in raw_sources:
            if isinstance(s, dict):
                if isinstance(s.get("page_content"), str):
                    ctx_texts.append(s["page_content"])
        if ctx_texts:
            has_any_context = True
        contexts.append(ctx_texts)
    except Exception as e:
        print(f"⚠️ Erreur API pour `{q}` -> {e}")
        answers.append("")
        contexts.append([])

    # 🔑 respecter limite : 1 requête / seconde
    time.sleep(1)

# ---------- HF Dataset ----------
hf_ds = Dataset.from_dict({
    "question": questions,
    "ground_truth": ground_truths,
    "answer": answers,
    "contexts": contexts,  # liste de listes de strings
})

# ---------- LLM & embeddings pour Ragas ----------
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
llm = MistralChatWrapper(client=mistral_client, model=EVAL_MODEL)
embeddings = MistralEmbeddings()

# ---------- Choix des métriques ----------
if has_any_context:
    metrics = [answer_relevancy, faithfulness, context_precision]
else:
    print("ℹ️ Aucun `page_content` dans /ask -> évaluation limitée à `answer_relevancy`.")
    metrics = [answer_relevancy]

# ---------- Evaluation (limiter la concurrence !) ----------
run_cfg = RunConfig(max_workers=1, timeout=120)  # crucial pour Mistral (1 req/s)

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
