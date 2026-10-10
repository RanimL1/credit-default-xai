from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression

DOSSIER_MODELES = Path(__file__).resolve().parent.parent / "models"

pipelines = joblib.load(DOSSIER_MODELES / "pipelines_opt.joblib")
X_train, _, _, _ = joblib.load(DOSSIER_MODELES / "split.joblib")
COLONNES = list(X_train.columns)

MODELES_SHAP = ["XGBoost", "Random Forest", "Régression logistique", "Arbre de décision"]
UNITE_LOG_ODDS = ["XGBoost", "Régression logistique"]
SEUIL_DEFAUT = 0.5

LIBELLES = {
    "PAY_1": ("le", "retard de paiement de septembre"),
    "PAY_2": ("le", "retard de paiement d'août"),
    "PAY_3": ("le", "retard de paiement de juillet"),
    "PAY_4": ("le", "retard de paiement de juin"),
    "PAY_5": ("le", "retard de paiement de mai"),
    "PAY_6": ("le", "retard de paiement d'avril"),
    "LIMIT_BAL": ("la", "limite de crédit"),
    "AGE": ("l'", "âge"),
    "SEX_2": ("le", "sexe"),
    "BILL_AMT1": ("le", "montant de la dernière facture"),
    "PAY_AMT1": ("le", "dernier paiement effectué"),
    "PAY_AMT2": ("l'", "avant-dernier paiement effectué"),
}
_DE = {"le": "du ", "la": "de la ", "l'": "de l'", "les": "des "}     # « à cause du retard »
_A = {"le": "au ", "la": "à la ", "l'": "à l'", "les": "aux "}        # « grâce au retard »


def _groupe(v, table):
    article, nom = LIBELLES.get(v, ("la", f"variable {v}"))
    return table[article] + nom


def modeles_disponibles():
    return list(pipelines.keys())


def obtenir_pipeline(nom):
    if nom not in pipelines:
        raise KeyError(nom)
    return pipelines[nom]


def vers_dataframe(client_dict):
    return pd.DataFrame([client_dict])[COLONNES]


@lru_cache(maxsize=None)
def noms_variables(nom_modele):
    prep = obtenir_pipeline(nom_modele).named_steps["prep"]
    return [n.split("__", 1)[1] for n in prep.get_feature_names_out()]


def transformer(nom_modele, X):
    prep = obtenir_pipeline(nom_modele).named_steps["prep"]
    Xt = prep.transform(X)
    return Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)


@lru_cache(maxsize=None)
def _arriere_plan(nom_modele):
    echantillon = X_train.sample(n=2000, random_state=42)
    return transformer(nom_modele, echantillon)


@lru_cache(maxsize=None)
def _explainer_shap(nom_modele):
    clf = obtenir_pipeline(nom_modele).named_steps["clf"]
    if isinstance(clf, LogisticRegression):
        return shap.LinearExplainer(clf, _arriere_plan(nom_modele))
    return shap.TreeExplainer(clf)


@lru_cache(maxsize=None)
def _explainer_lime(nom_modele):
    return LimeTabularExplainer(
        training_data=_arriere_plan(nom_modele),
        feature_names=noms_variables(nom_modele),
        class_names=["Pas de défaut", "Défaut"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )


def probabilite(nom_modele, client_dict):
    X = vers_dataframe(client_dict)
    return float(obtenir_pipeline(nom_modele).predict_proba(X)[0, 1])


def _contributions(noms, valeurs, client_dict, n):
    serie = pd.Series(valeurs, index=noms)
    top = serie.reindex(serie.abs().sort_values(ascending=False).index).head(n)
    return [
        {
            "variable": v,
            "valeur": float(client_dict[v]) if v in client_dict else None,
            "contribution": float(c),
        }
        for v, c in top.items()
    ]


def contributions_shap(nom_modele, client_dict, n=8):
    if nom_modele not in MODELES_SHAP:
        raise ValueError(f"SHAP n'est pas disponible pour '{nom_modele}'.")
    Xt = transformer(nom_modele, vers_dataframe(client_dict))
    ex = _explainer_shap(nom_modele)(Xt)
    vals = ex.values
    if vals.ndim == 3:                      # forêt / arbre : on garde la classe « défaut »
        vals = vals[0, :, 1]
    else:
        vals = vals[0]
    base = float(np.asarray(ex.base_values).reshape(-1)[-1])
    toutes = pd.Series(vals, index=noms_variables(nom_modele))
    return {
        "valeur_de_base": base,
        "unite": "log-odds" if nom_modele in UNITE_LOG_ODDS else "probabilité",
        "toutes": toutes,
        "contributions": _contributions(noms_variables(nom_modele), vals, client_dict, n),
    }


def contributions_lime(nom_modele, client_dict, n=8):
    clf = obtenir_pipeline(nom_modele).named_steps["clf"]
    Xt = transformer(nom_modele, vers_dataframe(client_dict))
    noms = noms_variables(nom_modele)
    exp = _explainer_lime(nom_modele).explain_instance(
        Xt[0], clf.predict_proba, num_features=n, num_samples=3000
    )
    paires = exp.as_map()[1]
    return _contributions(
        [noms[i] for i, _ in paires], [w for _, w in paires], client_dict, n
    )


def phrase_synthese(shap_vals, proba, seuil=SEUIL_DEFAUT, n=2):
    s = pd.Series(shap_vals)
    hausse = s[s > 0].sort_values(ascending=False).head(n).index.tolist()
    baisse = s[s < 0].sort_values().head(n).index.tolist()
    niveau = "élevé" if proba >= seuil else "faible"
    txt = f"Le risque de défaut est {niveau} ({proba:.0%})"
    if proba >= seuil and hausse:
        txt += " principalement à cause " + " et ".join(_groupe(v, _DE) for v in hausse)
    elif baisse:
        txt += " principalement grâce " + " et ".join(_groupe(v, _A) for v in baisse)
    return txt + "."
