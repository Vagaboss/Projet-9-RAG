import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import LLM
from pydantic import Field, ConfigDict
from mistralai import Mistral

from vector_pipe import MistralEmbeddings

# --- Charger variables d'environnement ---
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

# --- Charger FAISS + retriever ---
store_path = ROOT / "data" / "faiss_store"
embeddings = MistralEmbeddings()
db = FAISS.load_local(str(store_path), embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 10})

# --- Client Mistral pour génération ---
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("⚠️ La clé API Mistral n'est pas définie dans .env")
client = Mistral(api_key=api_key)
GEN_MODEL = "mistral-small-2503"  # tu peux changer en mistral-large-2411

# --- Wrapper LLM ---
class MistralChatWrapper(LLM):
    """Adapter le client Mistral chat à l’interface LLM de LangChain."""

    client: object = Field(...)
    model: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "mistral-chat"

    def _call(self, prompt: str, stop=None):
        messages = [
            {"role": "system", "content": "Tu es un assistant culturel. Réponds en français."},
            {"role": "user", "content": prompt},
        ]
        resp = self.client.chat.complete(model=self.model, messages=messages)
        return resp.choices[0].message.content.strip()

# --- Instancier LLM ---
llm = MistralChatWrapper(client=client, model=GEN_MODEL)

# --- Prompt personnalisé ---
prompt_template = """TTu es un assistant culturel qui recommande des événements uniquement à partir du CONTEXTE fourni ci-dessous.
Question : {question}

Contexte :
{context}

Consignes :
- Si le contexte contient des événements pertinents, donne un résumé clair avec :
  - titre(s) d’événement
  - date lisible
  - lieu
  - lien (si disponible)
- Si le contexte est vide ou ne contient aucun document pertinent, réponds exactement :
  "Désolé, aucun événement trouvé correspondant à ta recherche."
- N’invente jamais d’événements ou d’informations extérieures.
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "context"]
)

# --- Construire la chaîne RAG ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# --- Fonction réutilisable ---
def answer_question(question: str, k: int = 5):
    response = qa_chain.invoke(question)
    answer = response["result"]
    sources = response.get("source_documents", [])
    return answer, sources

# --- Interface CLI ---
if __name__ == "__main__":
    print("🤖 Chatbot culturel (RAG avec RetrievalQA) - tape 'quit' pour arrêter\n")
    while True:
        q = input("Vous: ")
        if q.lower() in {"quit", "exit"}:
            break
        response = qa_chain.invoke(q)
        answer = response["result"]
        print("\nAssistant:", answer)     
        print("\n---\n")





