import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
from sklearn.model_selection import train_test_split

st.title("2. Préparation")

if "df" not in st.session_state:
    st.warning("Charge d'abord les données (page 1).")
    st.stop()

df, cible, positif = st.session_state["df"], st.session_state["cible"], st.session_state["positif"]
taiwan = st.session_state.get("source") == "taiwan"

candidats = [c for c in df.columns if c != cible]
features = st.multiselect("Variables explicatives", candidats, default=candidats)

if taiwan:
    defaut_cat = [c for c in ["SEX", "EDUCATION", "MARRIAGE"] if c in features]
else:
    defaut_cat = [c for c in features if df[c].dtype == object or str(df[c].dtype) == "category"]
cat = st.multiselect("Variables catégorielles (encodage one-hot)", features, default=defaut_cat)
num = [c for c in features if c not in cat]

non_num = [c for c in num if not str(df[c].dtype).startswith(("int", "float", "uint"))]
if non_num:
    st.error(f"Ces variables ne sont pas numériques, déclare-les comme catégorielles : {non_num}")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    traitement = st.radio("Valeurs manquantes", ["Imputer", "Supprimer les lignes incomplètes"])
    imputation = st.radio("Imputation numérique", ["médiane", "moyenne"], horizontal=True,
                          disabled=traitement != "Imputer")
    normalisation = st.radio("Normalisation", ["Standard", "MinMax", "Aucune"], horizontal=True)
with c2:
    test_size = st.slider("Part du jeu de test", 0.10, 0.50, 0.20, 0.05)
    seed = st.number_input("Graine aléatoire", value=42, step=1)
    stratif = st.checkbox("Split stratifié (même proportion de la classe d'intérêt)", value=True)

if taiwan:
    st.caption("Pour que les explications de la page 5 (API) portent sur les mêmes clients, "
               "garde 0,20 et la graine 42.")

if st.button("Valider la préparation", type="primary"):
    if not features:
        st.error("Choisis au moins une variable.")
        st.stop()
    X, y = df[features].copy(), (df[cible] == positif).astype(int)
    if traitement != "Imputer":
        masque = X.notna().all(axis=1)
        X, y = X[masque], y[masque]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(seed), stratify=y if stratif else None)
    st.session_state.update(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        prep_cfg={"colonnes_cat": cat, "colonnes_num": num,
                  "imputation": imputation, "normalisation": normalisation},
    )
    for k in ("pipelines", "probas", "shap_cache"):
        st.session_state.pop(k, None)       # les anciens modèles ne correspondent plus
    st.success("Préparation validée. Passe à la page 3.")

if "X_train" in st.session_state:
    a, b, c = st.columns(3)
    a.metric("Train", len(st.session_state["X_train"]))
    b.metric("Test", len(st.session_state["X_test"]))
    c.metric("Classe d'intérêt (train / test)",
             f"{st.session_state['y_train'].mean():.1%} / {st.session_state['y_test'].mean():.1%}")
