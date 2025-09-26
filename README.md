# Projet 9 chatbot ia RAG

### 🎯 Objectif

Ce dépôt contient le Proof of Concept d’un chatbot RAG (Retrieval-Augmented Generation) pour recommander des événements culturels à partir de l’API OpenAgenda, en utilisant LangChain, Mistral et Faiss.

Ce document décrit uniquement l’installation de l’environnement de développement pour pouvoir exécuter les prochains scripts (ingestion, vectorisation, API).

### 🛠️ Prérequis

- Python ≥ 3.8 installé sur la machine (testé avec Python 3.13.3)

- Git installé

- Une connexion Internet pour télécharger les dépendances

- Système : Windows (fonctionne aussi sous Linux/Mac avec adaptations mineures)

### 📦 Installation
1. Cloner le projet

- git clone <url-du-repo>
- cd puls-events-rag

2. Créer un environnement virtuel

Selon ton terminal :

- PowerShell
python -m venv env
.\env\Scripts\Activate.ps1

- Invite de commandes (cmd.exe)
python -m venv env
.\env\Scripts\activate.bat

- Git Bash
python -m venv env
source env/Scripts/activate

⚠️ Le dossier env/ est ignoré par Git (cf. .gitignore).

3. Installer les dépendances

- pip install -r requirements.txt

- Les principales bibliothèques installées sont :

- faiss-cpu (vectorisation)

- langchain et langchain-community

- mistralai

- transformers, torch, sentence-transformers

- fastapi, uvicorn

- pytest, python-dotenv

4. Vérifier que l’environnement est bien activé

which python
ou
python -c "import sys; print(sys.prefix)"

Le chemin doit pointer vers le dossier du projet, par ex. :
.../Projet 9 RAG/env

5. Tester les imports

- Dans le shell Python :

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from mistralai.client import MistralClient

Si aucune erreur ne s’affiche ✅ → l’environnement est prêt.



📊 Résultats obtenus

Answer relevancy : 0.56
→ Dans un peu plus de la moitié des cas, la réponse est jugée pertinente par rapport à la question.
→ C’est “moyen”, mais logique pour un POC (on n’a pas encore optimisé le prompt, le retriever ou les embeddings).

Faithfulness : 0.59
→ La réponse respecte le contexte fourni dans ~60% des cas.
→ En clair : le bot a tendance à inventer ou extrapoler parfois.

Context precision : 0.10
→ Seulement 10% du contexte fourni est réellement utilisé pour générer la réponse.
→ Ça veut dire que ton retriever envoie beaucoup de “bruit” (docs non pertinents).

Context recall : 0.18
→ Seulement 18% des infos pertinentes du contexte sont utilisées.
→ Donc soit le retriever ne trouve pas toujours les bons passages, soit le modèle ne les exploite pas bien.