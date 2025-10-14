# Projet 9 chatbot ia RAG


## Lien github

https://github.com/Vagaboss/Projet-9-RAG 

.
## 🎯 Objectif

Ce projet a pour objectif de développer un système RAG (Retrieval-Augmented Generation) permettant de répondre en langage naturel à des questions sur des événements culturels et professionnels issus d’OpenAgenda.
L’utilisateur peut poser une question comme : 

 "Quels concerts de musique classique en avril 2025 à Paris ?".

Le système va alors :
- Rechercher les événements pertinents dans une base vectorielle construite avec FAISS.


- Générer une réponse claire et contextualisée grâce au modèle de langage Mistral.


- Fournir la réponse via une API REST exposée avec FastAPI.

## 📂 Structure du projet

api/main.py : API FastAPI avec les endpoints /ask, /rebuild et /health


rag/chatbot.py : Chaîne RAG qui combine FAISS + Mistral


rag/vector_pipe.py : Prétraitement des données et construction de l’index FAISS


scripts/build_index.py : Script pour reconstruire l’index FAISS


eval/eval_data.json : Jeu de test avec questions et réponses attendues


eval/evaluate_rag.py : Script d’évaluation avec Ragas


data/events_clean.json : Données sources nettoyées


requirements.txt : Dépendances Python


Dockerfile : Conteneurisation de l’API


README.md : Présentation du projet



### 📦 Installation
1. Cloner le projet

- git clone <https://github.com/Vagaboss/Projet-9-RAG.git>
- cd Projet-9-RAG

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

4. Définir la clé API Mistral
Créer un fichier .env à la racine du projet et ajouter :
 MISTRAL_API_KEY=ta_clef_api (recupéré sur le site de mistralai : prendre l'abonnement gratuit)

5. Construire l’index FAISS
python scripts/build_index.py

6. Lancer l’API FastAPI
uvicorn api.main:app --reload
Endpoints accessibles :
- Docs interactives : http://127.0.0.1:8000/docs
- Healthcheck : http://127.0.0.1:8000/health

7. Exemple d’appel API :

POST /ask
{ "question": "Quels concerts de musique classique en avril 2025 à Paris ?" }


## 🐳 Exécution avec Docker

1. Builder l’image
docker build -t rag-api .
2. Lancer le conteneur
docker run -p 8000:8000 --env-file .env rag-api
3. Accéder à l’API dans le docker
Swagger : http://127.0.0.1:8000/docs

## 📊 Évaluation avec Ragas
Verifier que l'api est bien lancée
Lancer l’évaluation
python -m eval.evaluate_rag
Exemple de résultats
Answer relevancy : 0.56


Faithfulness : 0.59


Context precision : 0.10


Context recall : 0.18


Ces résultats montrent que le système est pertinent mais qu’il peut encore être amélioré, notamment sur la couverture contextuelle.



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

## 👨‍💻 Auteur
Projet réalisé dans le cadre de la formation Data Scientist – OpenClassrooms.