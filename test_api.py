"""Test rapide de l'API (lance d'abord : uvicorn api.main:app --reload --port 8000)."""
import requests

URL = "http://localhost:8000"
client = {
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_1": 2, "PAY_2": 2, "PAY_3": 2, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 48000, "BILL_AMT2": 47000, "BILL_AMT3": 46000,
    "BILL_AMT4": 30000, "BILL_AMT5": 29000, "BILL_AMT6": 28000,
    "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 1500,
    "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 1000,
}

print("health :", requests.get(f"{URL}/health").json())
modeles = requests.get(f"{URL}/models").json()["modeles"]
print("modèles :", modeles)

for m in modeles:
    r = requests.post(f"{URL}/predict", json={"client": client, "modele": m, "seuil": 0.5}).json()
    print(f"{m:25s} proba = {r['probabilite_defaut']:.3f} -> {r['decision']}")

# Une entrée invalide doit être refusée (code 422)
r = requests.post(f"{URL}/predict", json={"client": {**client, "SEX": 5}})
print("SEX=5 ->", r.status_code, "(attendu : 422)")