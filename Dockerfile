# Étape 1 : choisir une image Python légère
FROM python:3.11-slim

# Étape 2 : définir le répertoire de travail
WORKDIR /app

# Étape 3 : copier les fichiers nécessaires
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Étape 4 : copier le code de ton projet
COPY . .

# Étape 5 : builder l’index FAISS au démarrage (optionnel, sinon tu peux le faire à la main)
#RUN python -m scripts.build_index || true

# Étape 6 : exposer le port de l’API
EXPOSE 8000

# Étape 7 : commande de lancement
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

