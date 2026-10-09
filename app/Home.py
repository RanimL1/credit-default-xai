import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

st.set_page_config(page_title="Classification du défaut de paiement", page_icon="📊", layout="wide")

st.title("📊 Classification du défaut de paiement")
st.markdown(
    """
Application interactive de **scoring de crédit** (Data Mining, ENSI).
Elle charge un jeu de données, entraîne et compare cinq modèles de classification.

**Parcours (menu de gauche) :**
1. **Données** : téléchargement automatique (UCI) ou import CSV, exploration
2. **Préparation** : variables, valeurs manquantes, encodage, normalisation, split
3. **Modèles** : régression logistique, arbre, kNN, Random Forest, XGBoost (+ optimisation)
4. **Évaluation** : métriques, courbes ROC/PR, matrice de confusion, curseur de seuil
"""
)

if "df" in st.session_state:
    st.info(f"Jeu de données chargé : {st.session_state['df'].shape[0]} lignes, "
            f"{st.session_state['df'].shape[1]} colonnes.")
