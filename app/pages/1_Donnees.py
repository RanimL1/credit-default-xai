import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from core.data import charger_csv, charger_taiwan

st.title("1. Données")


@st.cache_data(show_spinner="Téléchargement du jeu de données (UCI)...")
def get_taiwan():
    return charger_taiwan()


source = st.radio("Source des données",
                  ["Télécharger le jeu Taiwan Credit Default (UCI)", "Importer mon fichier CSV"])

if source.startswith("Télécharger"):
    if st.button("Charger le jeu de données", type="primary") or st.session_state.get("source") == "taiwan":
        st.session_state.update(df=get_taiwan(), cible="DEFAULT", positif=1, source="taiwan")
else:
    fichier = st.file_uploader("Fichier CSV", type="csv")
    if fichier is not None:
        df_csv = charger_csv(fichier)
        cible = st.selectbox("Variable cible", df_csv.columns)
        classes = sorted(df_csv[cible].dropna().unique().tolist(), key=str)
        if len(classes) != 2:
            st.error(f"La cible doit être binaire ({len(classes)} valeurs trouvées). "
                     "Ramène-la à deux classes avant l'import.")
        else:
            positif = st.selectbox("Classe d'intérêt (la classe à prédire)", classes, index=1)
            if st.button("Valider", type="primary"):
                st.session_state.update(df=df_csv, cible=cible, positif=positif, source="csv")

if "df" not in st.session_state:
    st.info("Charge un jeu de données pour commencer.")
    st.stop()

df, cible, positif = st.session_state["df"], st.session_state["cible"], st.session_state["positif"]
y01 = (df[cible] == positif).astype(int)

c1, c2, c3 = st.columns(3)
c1.metric("Observations", f"{len(df):,}")
c2.metric("Variables", df.shape[1] - 1)
c3.metric("Classe d'intérêt", f"{y01.mean():.1%}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Aperçu", "Statistiques", "Valeurs manquantes", "Classes", "Exploration"])

with tab1:
    st.dataframe(df.head(50), width="stretch")
    st.write("Types :", df.dtypes.astype(str).value_counts().to_dict())

with tab2:
    st.dataframe(df.describe().T.round(2), width="stretch")

with tab3:
    manq = df.isna().sum()
    manq = manq[manq > 0]
    if manq.empty:
        st.success("Aucune valeur manquante.")
    else:
        st.dataframe(manq.rename("Manquantes").to_frame())

with tab4:
    comptes = y01.value_counts().rename({0: "Autre classe", 1: "Classe d'intérêt"})
    st.plotly_chart(px.bar(x=comptes.index, y=comptes.values, labels={"x": "Classe", "y": "Effectif"},
                           title="Répartition des classes"), width="stretch")
    if y01.mean() < 0.35 or y01.mean() > 0.65:
        st.warning("Classes déséquilibrées : privilégie PR-AUC et rappel plutôt que l'accuracy, "
                   "et compense avec les poids de classes.")

with tab5:
    variables = [c for c in df.columns if c != cible]
    v = st.selectbox("Variable à explorer", variables)
    numerique = pd.api.types.is_numeric_dtype(df[v])
    g1, g2 = st.columns(2)
    with g1:
        if numerique:
            fig = px.histogram(df.assign(_c=y01.astype(str)), x=v, color="_c", barmode="overlay",
                               opacity=0.65, title=f"Distribution de {v} selon la classe")
        else:
            fig = px.histogram(df.assign(_c=y01.astype(str)), x=v, color="_c", barmode="group",
                               title=f"Distribution de {v} selon la classe")
        st.plotly_chart(fig, width="stretch")
    with g2:
        groupe = pd.qcut(df[v], 10, duplicates="drop").astype(str) if numerique and df[v].nunique() > 10 else df[v].astype(str)
        taux = y01.groupby(groupe).mean().reset_index()
        taux.columns = [v, "Taux classe d'intérêt"]
        st.plotly_chart(px.bar(taux, x=v, y="Taux classe d'intérêt",
                               title=f"Taux de la classe d'intérêt par {v}"), width="stretch")
    num = df.select_dtypes("number")
    if num.shape[1] > 1:
        st.plotly_chart(px.imshow(num.corr().round(2), color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                  title="Corrélations", aspect="auto"), width="stretch")
