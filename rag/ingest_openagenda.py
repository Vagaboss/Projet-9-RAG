import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

BASE_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

# --- Nettoyage texte ---
def clean_html(raw_html):
    """Nettoie le HTML et retourne du texte brut"""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def clean_text(text):
    """Supprime caractères invisibles (LS, PS) et normalise les espaces"""
    if not text:
        return ""
    return (
        text.replace("\u2028", " ")
            .replace("\u2029", " ")
            .replace("\xa0", " ")  # espace insécable
            .strip()
    )

# --- API OpenAgenda ---
def get_total(city="Paris", year="2025"):
    """Récupère le nombre total d'événements pour une ville et une année (par update)"""
    params = {
        "refine": [
            f"location_city:\"{city}\"",
            f"updatedat:\"{year}\""
        ],
        "limit": 0
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    data = response.json()
    return data["total_count"]

def fetch_events(city="Paris", year="2025", limit=100, offset=0):
    """Récupère une page d'événements"""
    params = {
        "refine": [
            f"location_city:\"{city}\"",
            f"updatedat:\"{year}\""
        ],
        "limit": limit,
        "offset": offset
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()

def extract_records(json_data):
    """Extrait et nettoie les champs utiles d'une réponse JSON"""
    records = []
    for rec in json_data.get("results", []):
        records.append({
            "id": rec.get("uid"),
            "title": clean_text(rec.get("title_fr")),
            "description": clean_text(rec.get("description_fr")),
            "long_description": clean_text(clean_html(rec.get("longdescription_fr"))),
            "keywords": rec.get("keywords_fr"),
            "city": rec.get("location_city"),
            "region": rec.get("location_region"),
            "country": rec.get("location_countrycode"),
            "address": clean_text(rec.get("location_address")),
            "coordinates": rec.get("location_coordinates"),
            "date_start": rec.get("firstdate_begin"),
            "date_end": rec.get("lastdate_end"),
            "url": rec.get("canonicalurl")
        })
    return records

# --- Script principal ---
if __name__ == "__main__":
    city = "Paris"
    year = "2025"
    step = 100

    # Dossier data
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Nombre total d'événements
    total = get_total(city, year)
    print(f"Nombre total d'événements trouvés (maj en {year}) pour {city} : {total}")

    # 2. Récupération paginée
    all_records = []
    for offset in range(0, total, step):
        print(f"Récupération des événements {offset} à {offset+step}...")
        data = fetch_events(city=city, year=year, limit=step, offset=offset)
        all_records.extend(extract_records(data))

    # 3. Conversion DataFrame + nettoyage Pandas
    df = pd.DataFrame(all_records)

    # Conversion des dates
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")

    # Filtrer uniquement les événements qui commencent en 2025
    df = df[df["date_start"].dt.year == 2025]

    # Gestion des valeurs manquantes
    df["title"] = df["title"].fillna("Titre manquant")
    df["description"] = df["description"].fillna("Description manquante")
    df["long_description"] = df["long_description"].fillna(df["description"])
    df["region"] = df["region"].fillna("Île-de-France")
    df["keywords"] = df["keywords"].fillna("[]")

    # 4. Sauvegarde CSV + JSON
    out_csv = DATA_DIR / "events_clean.csv"
    out_json = DATA_DIR / "events_clean.json"

    df.to_csv(out_csv, index=False, encoding="utf-8")
    df.to_json(out_json, orient="records", force_ascii=False, indent=2)

    print(f"{len(df)} événements filtrés (date_start=2025) sauvegardés dans {out_csv} et {out_json}")


