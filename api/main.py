from fastapi import FastAPI, HTTPException

from api import modeles
from api.schemas import Prediction, Requete

app = FastAPI(
    title="API de scoring - Défaut de paiement (Taiwan)",
    description="Prédit la probabilité de défaut d'un client avec le modèle choisi.",
    version="1.0.0",
)


@app.get("/health", tags=["Service"])
def health():
    return {"status": "ok"}


@app.get("/models", tags=["Service"])
def models():
    return {"modeles": modeles.modeles_disponibles(), "variables": modeles.COLONNES}


@app.post("/predict", response_model=Prediction, tags=["Prédiction"])
def predict(req: Requete):
    if req.modele not in modeles.modeles_disponibles():
        raise HTTPException(
            status_code=404,
            detail=f"Modèle inconnu : '{req.modele}'. Disponibles : {modeles.modeles_disponibles()}",
        )
    p = modeles.probabilite_defaut(req.modele, req.client.model_dump())
    return Prediction(
        modele=req.modele,
        probabilite_defaut=p,
        seuil=req.seuil,
        decision="defaut_probable" if p >= req.seuil else "pas_de_defaut",
    )