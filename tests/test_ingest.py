import os
import pandas as pd

def test_ingestion_outputs():
    """Vérifie que les fichiers CSV et JSON sont bien générés et non vides"""
    csv_path = "data/events_clean.csv"
    json_path = "data/events_clean.json"

    # Fichiers existent
    assert os.path.exists(csv_path), f"{csv_path} est introuvable"
    assert os.path.exists(json_path), f"{json_path} est introuvable"

    # Chargement CSV
    df = pd.read_csv(csv_path)

    # Non vide
    assert len(df) > 0, "Le fichier CSV est vide"

    # Colonnes attendues
    expected_cols = {
        "id", "title", "description", "long_description", "keywords",
        "city", "region", "country", "address", "coordinates",
        "date_start", "date_end", "url"
    }
    assert expected_cols.issubset(df.columns), "Colonnes manquantes dans le CSV"

    # Tous les événements doivent être à Paris
    assert (df["city"] == "Paris").all(), "Certains événements ne sont pas à Paris"

    # Toutes les dates de début doivent être en 2025
    years = pd.to_datetime(df["date_start"], errors="coerce").dt.year.dropna().unique()
    assert (years == 2025).all(), f"Des événements n'appartiennent pas à l'année 2025 : {years}"
