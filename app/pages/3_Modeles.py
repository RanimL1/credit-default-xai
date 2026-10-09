import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from core.metrics import tableau_comparatif
from core.models import GRILLES, NOMS_MODELES, creer_modele
from core.preprocess import construire_preprocesseur

st.title("3. Modélisation")

if "X_train" not in st.session_state:
    st.warning("Valide d'abord la préparation (page 2).")
    st.stop()

X_train, X_test = st.session_state["X_train"], st.session_state["X_test"]
y_train, y_test = st.session_state["y_train"], st.session_state["y_test"]
cfg = st.session_state["prep_cfg"]
ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))


def parametres(nom):
    k = lambda p: f"{nom}_{p}"
    if nom == "Régression logistique":
        return {"C": st.select_slider("C (régularisation inverse)", [0.001, 0.01, 0.1, 1, 10, 100], 1, key=k("C"))}
    if nom == "Arbre de décision":
        return {"max_depth": st.slider("Profondeur max", 2, 15, 5, key=k("d")),
                "min_samples_leaf": st.slider("Min. échantillons par feuille", 1, 200, 20, key=k("l"))}
    if nom == "k plus proches voisins":
        return {"n_neighbors": st.slider("Nombre de voisins", 3, 100, 25, key=k("n")),
                "weights": st.selectbox("Pondération", ["uniform", "distance"], key=k("w"))}
    if nom == "Random Forest":
        return {"n_estimators": st.slider("Nombre d'arbres", 100, 500, 300, 100, key=k("n")),
                "max_depth": st.slider("Profondeur max", 3, 20, 8, key=k("d")),
                "min_samples_leaf": st.slider("Min. échantillons par feuille", 1, 50, 5, key=k("l"))}
    return {"n_estimators": st.slider("Nombre d'arbres", 100, 600, 300, 100, key=k("n")),
            "max_depth": st.slider("Profondeur max", 2, 10, 4, key=k("d")),
            "learning_rate": st.select_slider("Taux d'apprentissage", [0.01, 0.03, 0.05, 0.1, 0.2], 0.05, key=k("lr"))}


choix = st.multiselect("Modèles à entraîner", NOMS_MODELES, default=NOMS_MODELES)
poids = st.checkbox("Compenser le déséquilibre (poids de classes, train uniquement)", value=True)
optim = st.checkbox("Optimiser les hyperparamètres (RandomizedSearchCV, validation croisée sur le train)")
n_iter = st.slider("Combinaisons testées par modèle", 4, 30, 10, disabled=not optim)

params_choisis = {}
if not optim:
    for nom in choix:
        with st.expander(f"Hyperparamètres : {nom}"):
            params_choisis[nom] = parametres(nom)
else:
    st.caption("Les grilles de recherche sont définies dans core/models.py. "
               "Attention : Random Forest et XGBoost peuvent prendre plusieurs minutes.")

if st.button("Entraîner", type="primary", disabled=not choix):
    pipelines, probas, finaux = {}, {}, {}
    barre = st.progress(0.0)
    for i, nom in enumerate(choix):
        with st.spinner(f"Entraînement : {nom}"):
            prep = construire_preprocesseur(**cfg)
            if optim:
                grille = GRILLES[nom]
                taille = 1
                for v in grille.values():
                    taille *= len(v)
                rech = RandomizedSearchCV(
                    Pipeline([("prep", prep), ("clf", creer_modele(nom, {}, ratio, poids))]),
                    grille, n_iter=min(n_iter, taille), scoring="roc_auc",
                    cv=StratifiedKFold(3, shuffle=True, random_state=42),
                    n_jobs=-1, random_state=42)
                rech.fit(X_train, y_train)
                pipe = rech.best_estimator_
                finaux[nom] = {k.replace("clf__", ""): v for k, v in rech.best_params_.items()}
            else:
                pipe = Pipeline([("prep", prep), ("clf", creer_modele(nom, params_choisis[nom], ratio, poids))])
                pipe.fit(X_train, y_train)
                finaux[nom] = params_choisis[nom]
            pipelines[nom] = pipe
            probas[nom] = pipe.predict_proba(X_test)[:, 1]
        barre.progress((i + 1) / len(choix))
    st.session_state.update(pipelines=pipelines, probas=probas, params_finaux=finaux)
    st.session_state.pop("shap_cache", None)
    st.success("Modèles entraînés.")

if "pipelines" in st.session_state:
    st.subheader("Tableau comparatif (test, seuil 0,5)")
    st.dataframe(tableau_comparatif(y_test, st.session_state["probas"]), width="stretch")
    with st.expander("Hyperparamètres retenus"):
        st.json(st.session_state["params_finaux"])
