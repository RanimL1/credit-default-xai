import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import joblib
import streamlit as st
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
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

RACINE = Path(__file__).resolve().parents[2]
CHEMIN_OPT = RACINE / "models" / "pipelines_opt.joblib"
CHEMIN_SPLIT = RACINE / "models" / "split.joblib"
ECHANTILLON_RECHERCHE = 10_000   # taille max de l'échantillon du train utilisé pour la recherche
taiwan = st.session_state.get("source") == "taiwan"


def _py(v):
    return v.item() if hasattr(v, "item") else v


if taiwan:
    st.subheader("Option rapide : modèles déjà optimisés")
    st.caption("Les cinq modèles ont été optimisés avec RandomizedSearchCV dans le notebook. "
               "Les charger évite de relancer la recherche, trop lourde pour l'hébergement gratuit. "
               "Valable pour le jeu Taiwan avec le split de la page 2 réglé sur 0,20 et la graine 42.")
    if st.button("Charger les modèles optimisés", disabled=not (CHEMIN_OPT.exists() and CHEMIN_SPLIT.exists())):
        try:
            ref = joblib.load(CHEMIN_SPLIT)
            pre = {n: p for n, p in joblib.load(CHEMIN_OPT).items() if n in NOMS_MODELES}
            # Même jeu de test que celui des modèles : sinon des clients du test auraient servi à l'entraînement.
            if set(X_test.index) != set(ref[1].index):
                st.error("Le split de la page 2 ne correspond pas à celui des modèles pré-entraînés "
                         "(jeu Taiwan, test 0,20, graine 42). Reviens à la page 2, choisis ces paramètres "
                         "et valide à nouveau.")
            else:
                probas = {n: p.predict_proba(X_test)[:, 1] for n, p in pre.items()}
                finaux = {n: {k.replace("clf__", ""): _py(pre[n].named_steps["clf"].get_params()[k.replace("clf__", "")])
                              for k in GRILLES[n]} for n in pre}
                st.session_state.update(pipelines=pre, probas=probas, params_finaux=finaux)
                st.success("Modèles optimisés chargés.")
        except Exception as e:
            st.error(f"Chargement impossible : {type(e).__name__}: {e}")

    st.divider()
    st.subheader("Ou entraîner les modèles ici")


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
    taille_txt = f"{ECHANTILLON_RECHERCHE:,}".replace(",", " ")
    st.caption(f"Les grilles de recherche sont définies dans core/models.py. La recherche porte sur un "
               f"échantillon d'au plus {taille_txt} lignes du train ; le meilleur modèle est "
               f"ensuite réentraîné sur tout le train. Random Forest et XGBoost restent les plus lourds : "
               f"sur l'hébergement gratuit, limite le nombre de combinaisons.")

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
                if len(X_train) > ECHANTILLON_RECHERCHE:
                    X_rech, _, y_rech, _ = train_test_split(
                        X_train, y_train, train_size=ECHANTILLON_RECHERCHE,
                        stratify=y_train, random_state=42)
                else:
                    X_rech, y_rech = X_train, y_train
                rech = RandomizedSearchCV(
                    Pipeline([("prep", prep), ("clf", creer_modele(nom, {}, ratio, poids, n_jobs=1))]),
                    grille, n_iter=min(n_iter, taille), scoring="roc_auc",
                    cv=StratifiedKFold(3, shuffle=True, random_state=42),
                    n_jobs=2, random_state=42)
                rech.fit(X_rech, y_rech)
                pipe = clone(rech.best_estimator_)
                pipe.fit(X_train, y_train)          # meilleur modèle réentraîné sur tout le train
                finaux[nom] = {k.replace("clf__", ""): v for k, v in rech.best_params_.items()}
            else:
                pipe = Pipeline([("prep", prep), ("clf", creer_modele(nom, params_choisis[nom], ratio, poids))])
                pipe.fit(X_train, y_train)
                finaux[nom] = params_choisis[nom]
            pipelines[nom] = pipe
            probas[nom] = pipe.predict_proba(X_test)[:, 1]
        barre.progress((i + 1) / len(choix))
    st.session_state.update(pipelines=pipelines, probas=probas, params_finaux=finaux)
    st.success("Modèles entraînés.")

if "pipelines" in st.session_state:
    st.subheader("Tableau comparatif (test, seuil 0,5)")
    st.dataframe(tableau_comparatif(y_test, st.session_state["probas"]), width="stretch")
    with st.expander("Hyperparamètres retenus"):
        st.json(st.session_state["params_finaux"])