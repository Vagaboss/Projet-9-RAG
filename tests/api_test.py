import requests

API_URL = "http://127.0.0.1:8000"

def test_health():
    """Test du endpoint health"""
    r = requests.get(f"{API_URL}/health")
    print("🔹 /health ->", r.status_code, r.json())

def test_ask():
    """Test du endpoint ask"""
    question = {"question": "concert de musique classique en avril 2025 à Paris"}
    r = requests.post(f"{API_URL}/ask", json=question)
    print("🔹 /ask ->", r.status_code)
    print(r.json())

def test_rebuild():
    """Test du endpoint rebuild"""
    r = requests.post(f"{API_URL}/rebuild")
    print("🔹 /rebuild ->", r.status_code, r.json())

if __name__ == "__main__":
    print("=== Tests API ===")
    test_health()
    test_ask()
    test_rebuild()
    # ⚠️ Ne lance rebuild que si tu veux vraiment réindexer (ça prend du temps !)
    # test_rebuild()
