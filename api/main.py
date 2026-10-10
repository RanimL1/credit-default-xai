from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from api import explain
from api.schemas import (
    Prediction, ReponseComplete, ReponseLime, ReponseShap, Requete,
)

app = FastAPI(
    title="API d'interprétation - Défaut de paiement (Taiwan)",
    description=(
        "Prédit la probabilité de défaut d'un client et explique la décision "
        "avec SHAP et LIME. Contribution > 0 : augmente le risque ; < 0 : le diminue."
    ),
    version="1.0.0",
)
@app.get("/", include_in_schema=False)
def racine():
       return RedirectResponse(url="/docs")

def _verifier_modele(nom):
    if nom not in explain.modeles_disponibles():
        raise HTTPException(
            status_code=404,
            detail=f"Modèle inconnu : '{nom}'. Disponibles : {explain.modeles_disponibles()}",
        )


@app.get("/health", tags=["Service"])
def health():
    return {"status": "ok"}


@app.get("/models", tags=["Service"])
def models():
    return {
        "modeles": explain.modeles_disponibles(),
        "shap_disponible": explain.MODELES_SHAP,
        "variables": explain.COLONNES,
    }


@app.post("/predict", response_model=Prediction, tags=["Prédiction"])
def predict(req: Requete):
    _verifier_modele(req.modele)
    p = explain.probabilite(req.modele, req.client.model_dump())
    return Prediction(
        modele=req.modele,
        probabilite_defaut=p,
        seuil=req.seuil,
        decision="defaut_probable" if p >= req.seuil else "pas_de_defaut",
    )


@app.post("/explain/shap", response_model=ReponseShap, tags=["Explication"])
def explain_shap(req: Requete):
    _verifier_modele(req.modele)
    client = req.client.model_dump()
    try:
        res = explain.contributions_shap(req.modele, client, req.n_variables)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    p = explain.probabilite(req.modele, client)
    return ReponseShap(
        modele=req.modele,
        probabilite_defaut=p,
        valeur_de_base=res["valeur_de_base"],
        unite=res["unite"],
        contributions=res["contributions"],
        phrase=explain.phrase_synthese(res["toutes"], p, req.seuil),
    )


@app.post("/explain/lime", response_model=ReponseLime, tags=["Explication"])
def explain_lime(req: Requete):
    _verifier_modele(req.modele)
    client = req.client.model_dump()
    return ReponseLime(
        modele=req.modele,
        probabilite_defaut=explain.probabilite(req.modele, client),
        contributions=explain.contributions_lime(req.modele, client, req.n_variables),
    )


@app.post("/explain", response_model=ReponseComplete, tags=["Explication"])
def explain_complet(req: Requete):
    """Prédiction + SHAP (si disponible) + LIME + comparaison + phrase de synthèse."""
    _verifier_modele(req.modele)
    client = req.client.model_dump()
    p = explain.probabilite(req.modele, client)
    prediction = Prediction(
        modele=req.modele,
        probabilite_defaut=p,
        seuil=req.seuil,
        decision="defaut_probable" if p >= req.seuil else "pas_de_defaut",
    )
    lime = ReponseLime(
        modele=req.modele,
        probabilite_defaut=p,
        contributions=explain.contributions_lime(req.modele, client, max(req.n_variables, 5)),
    )

    shap_rep, phrase, communes = None, "", []
    if req.modele in explain.MODELES_SHAP:
        res = explain.contributions_shap(req.modele, client, max(req.n_variables, 5))
        phrase = explain.phrase_synthese(res["toutes"], p, req.seuil)
        shap_rep = ReponseShap(
            modele=req.modele,
            probabilite_defaut=p,
            valeur_de_base=res["valeur_de_base"],
            unite=res["unite"],
            contributions=res["contributions"][: req.n_variables],
            phrase=phrase,
        )
        top_shap = {c["variable"] for c in res["contributions"][:5]}
        top_lime = {c.variable for c in lime.contributions[:5]}
        communes = sorted(top_shap & top_lime)
    else:
        phrase = "SHAP n'est pas disponible pour ce modèle ; seule l'explication LIME est fournie."

    lime.contributions = lime.contributions[: req.n_variables]
    return ReponseComplete(
        prediction=prediction,
        shap=shap_rep,
        lime=lime,
        variables_communes_top5=communes,
        phrase=phrase,
    )
