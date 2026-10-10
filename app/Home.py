import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core.api_client import api_disponible, url_api

st.set_page_config(page_title="Classification explicable", page_icon="📊", layout="wide")

st.title("📊 Classification explicable du défaut de paiement")
st.markdown(
    """
Application interactive de **scoring de crédit explicable** (Data Mining, ENSI).
Elle charge un jeu de données, entraîne et compare cinq modèles, puis explique leurs décisions
avec **SHAP** et **LIME**, globalement et client par client.

**Parcours (menu de gauche) :**
1. **Données** : téléchargement automatique (UCI) ou import CSV, exploration
2. **Préparation** : variables, valeurs manquantes, encodage, normalisation, split
3. **Modèles** : régression logistique, arbre, kNN, Random Forest, XGBoost (+ optimisation)
4. **Évaluation** : métriques, courbes ROC/PR, matrice de confusion, curseur de seuil
5. **Explications (XAI)** : importances, SHAP, LIME, phrase de synthèse, équité
"""
)

with st.sidebar:
    st.subheader("API d'interprétation")
    if api_disponible():
        st.success("API connectée")
    else:
        st.warning("API injoignable : les explications locales (page 5) seront indisponibles.")
    st.caption(url_api())

if "df" in st.session_state:
    st.info(f"Jeu de données chargé : {st.session_state['df'].shape[0]} lignes, "
            f"{st.session_state['df'].shape[1]} colonnes.")
