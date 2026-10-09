from pathlib import Path

import joblib
import pandas as pd

DOSSIER_MODELES = Path(__file__).resolve().parent.parent / "models"

pipelines = joblib.load(DOSSIER_MODELES / "pipelines_opt.joblib")

_premier = next(iter(pipelines.values()))
COLONNES = list(_premier.named_steps["prep"].feature_names_in_)


def modeles_disponibles():
    return list(pipelines.keys())


def probabilite_defaut(nom_modele, client_dict):
    X = pd.DataFrame([client_dict])[COLONNES]
    return float(pipelines[nom_modele].predict_proba(X)[0, 1])