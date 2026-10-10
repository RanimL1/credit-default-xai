import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import recall_score

from core.api_client import api_disponible, expliquer, liste_modeles
from core.xai import calcul_shap, importance_native, importance_permutation

st.title("5. Explications (XAI)")

if "pipelines" not in st.session_state:
    st.warning("Entraîne d'abord des modèles (page 3).")
    st.stop()

pipelines = st.session_state["pipelines"]
X_train, X_test = st.session_state["X_train"], st.session_state["X_test"]
y_test = st.session_state["y_test"]
seuil_global = float(st.session_state.get("seuil", 0.5))

onglet_g, onglet_l, onglet_e = st.tabs(["Explications globales", "Explication d'un client (API)", "Équité"])

# ------------------------------------------------------------------ GLOBAL
with onglet_g:
    nom = st.selectbox("Modèle à expliquer", list(pipelines), key="xai_modele")
    pipe = pipelines[nom]

    st.subheader("Importance propre au modèle")
    imp = importance_native(pipe)
    if imp is None:
        st.info("Ce modèle n'a pas d'importance native (voir la permutation ci-dessous).")
    else:
        top = imp.reindex(imp.abs().sort_values(ascending=False).index).head(15).iloc[::-1]
        titre = "Coefficients (variables normalisées)" if hasattr(pipe.named_steps["clf"], "coef_") else "Importance (impureté / gain)"
        st.plotly_chart(px.bar(x=top.values, y=top.index, orientation="h", title=titre,
                               labels={"x": "", "y": ""}), width="stretch")

    st.subheader("Importance par permutation (comparable entre modèles)")
    if st.button("Calculer la permutation pour tous les modèles"):
        with st.spinner("Calcul en cours..."):
            st.session_state["perm"] = pd.DataFrame(
                {n: importance_permutation(p, X_test, y_test) for n, p in pipelines.items()})
    if "perm" in st.session_state:
        perm = st.session_state["perm"]
        top = perm.mean(axis=1).sort_values(ascending=False).head(12).index
        fig = px.bar(perm.loc[top].iloc[::-1].reset_index().melt(id_vars="index"),
                     x="value", y="index", color="variable", barmode="group", orientation="h",
                     labels={"value": "Baisse d'AUC", "index": "", "variable": "Modèle"})
        st.plotly_chart(fig, width="stretch")

    st.subheader("SHAP : poids et sens des variables")
    n_ech = st.slider("Taille de l'échantillon de test", 200, 1500, 500, 100)
    if st.button("Calculer SHAP"):
        try:
            with st.spinner("Calcul des valeurs SHAP..."):
                echantillon = X_test.sample(min(n_ech, len(X_test)), random_state=42)
                st.session_state.setdefault("shap_cache", {})[nom] = calcul_shap(pipe, X_train, echantillon)
        except ValueError as e:
            st.error(str(e))
    expl = st.session_state.get("shap_cache", {}).get(nom)
    if expl is not None:
        shap.plots.beeswarm(expl, max_display=15, show=False)
        fig = plt.gcf()
        st.pyplot(fig, clear_figure=True)
        plt.close("all")

        moyens = pd.Series(np.abs(expl.values).mean(axis=0), index=expl.feature_names).sort_values(ascending=False)
        st.caption("Une valeur SHAP positive pousse la prédiction vers la classe d'intérêt. "
                   "Pour XGBoost et la régression logistique, l'unité est le log-odds.")
        var = st.selectbox("Graphique de dépendance", moyens.head(10).index.tolist())
        shap.plots.scatter(expl[:, var], show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        plt.close("all")

# ------------------------------------------------------------------ LOCAL (API)
EXEMPLE = {
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_1": 2, "PAY_2": 2, "PAY_3": 2, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 48000, "BILL_AMT2": 47000, "BILL_AMT3": 46000,
    "BILL_AMT4": 30000, "BILL_AMT5": 29000, "BILL_AMT6": 28000,
    "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 1500, "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 1000,
}


def en_natif(d):
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}


def graphique_waterfall(rep):
    """Cascade des contributions SHAP (les variables non affichées sont regroupées)."""
    s = rep["shap"]
    base, unite, p = s["valeur_de_base"], s["unite"], rep["prediction"]["probabilite_defaut"]
    final = float(np.log(p / (1 - p))) if unite == "log-odds" else p
    contribs = s["contributions"]
    autres = final - base - sum(c["contribution"] for c in contribs)
    etiquettes = [f"{c['variable']} = {c['valeur']:g}" if c["valeur"] is not None else c["variable"]
                  for c in contribs]
    x = ["Valeur de base"] + etiquettes + ["Autres variables", "Prédiction"]
    y = [base] + [c["contribution"] for c in contribs] + [autres, final]
    mesure = ["absolute"] + ["relative"] * (len(contribs) + 1) + ["total"]
    fig = go.Figure(go.Waterfall(x=x, y=y, measure=mesure,
                                 increasing={"marker": {"color": "#d62728"}},
                                 decreasing={"marker": {"color": "#1f77b4"}}))
    fig.update_layout(title="SHAP waterfall : rouge = augmente le risque, bleu = le diminue",
                      yaxis_title=unite, xaxis_tickangle=-35)
    return fig


def onglet_local():
        if st.session_state.get("source") != "taiwan":
            st.info("L'API d'interprétation est entraînée sur le jeu Taiwan Credit Default. "
                    "Charge ce jeu (page 1) pour utiliser l'explication individuelle.")
            return
        if not api_disponible():
            st.error("L'API est injoignable. Lance-la (uvicorn api.main:app --port 8000) "
                     "ou vérifie API_URL dans les secrets.")
            return
        try:
            modeles_api = liste_modeles()["modeles"]
        except Exception:
            modeles_api = ["XGBoost", "Random Forest", "Régression logistique", "Arbre de décision"]

        cA, cB = st.columns(2)
        modele_api = cA.selectbox("Modèle (côté API)", modeles_api)
        seuil = cB.slider("Seuil de décision", 0.05, 0.95, seuil_global, 0.01, key="seuil_local")
        mode = st.radio("Client", ["Sélectionner dans le jeu de test", "Saisir un client fictif"], horizontal=True)

        if mode.startswith("Sélectionner"):
            pos = st.number_input("Position dans le jeu de test", 0, len(X_test) - 1, 0, 1)
            ligne = X_test.iloc[int(pos)]
            client = en_natif(ligne.to_dict())
            st.caption(f"Défaut réel observé : {int(y_test.iloc[int(pos)])}")
            st.dataframe(ligne.to_frame("Valeur").T, width="stretch")
        else:
            client, colonnes = {}, st.columns(3)
            for i, (k, v) in enumerate(EXEMPLE.items()):
                with colonnes[i % 3]:
                    client[k] = st.number_input(k, value=v, step=1 if isinstance(v, int) else 100.0)

        manquantes = [k for k in EXEMPLE if k not in client]
        if manquantes:
            st.error(f"Il manque des variables pour l'API : {manquantes}. Garde les 23 variables à la page 2.")
            return

        if st.button("Expliquer ce client", type="primary"):
            try:
                with st.spinner("Appel de l'API (la première requête peut prendre un moment)..."):
                    st.session_state["rep_api"] = expliquer(client, modele_api, seuil, 8)
            except Exception as e:
                st.error(f"Erreur de l'API : {e}")

        rep = st.session_state.get("rep_api")
        if rep:
            pr = rep["prediction"]
            a, b = st.columns(2)
            a.metric("Probabilité de défaut", f"{pr['probabilite_defaut']:.1%}")
            b.metric("Décision au seuil choisi",
                     "Défaut probable" if pr["probabilite_defaut"] >= seuil else "Pas de défaut")
            st.info(rep["phrase"])

            if rep.get("shap"):
                st.plotly_chart(graphique_waterfall(rep), width="stretch")

            lime = pd.DataFrame(rep["lime"]["contributions"])
            fig = px.bar(lime.iloc[::-1], x="contribution", y="variable", orientation="h",
                         color=lime.iloc[::-1]["contribution"] > 0,
                         color_discrete_map={True: "#d62728", False: "#1f77b4"},
                         title="Explication LIME du même client")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

            if rep.get("shap"):
                comp = pd.DataFrame(rep["shap"]["contributions"])[["variable", "contribution"]].rename(
                    columns={"contribution": "SHAP"}).merge(
                    lime[["variable", "contribution"]].rename(columns={"contribution": "LIME"}),
                    on="variable", how="outer")
                comp["Même sens ?"] = np.where(comp[["SHAP", "LIME"]].notna().all(axis=1),
                                               np.sign(comp["SHAP"]) == np.sign(comp["LIME"]), None)
                st.subheader("Comparaison SHAP / LIME")
                st.dataframe(comp.round(3), width="stretch")
                st.write("Variables communes au top 5 :", ", ".join(rep["variables_communes_top5"]) or "aucune")


with onglet_l:
    onglet_local()

# ------------------------------------------------------------------ ÉQUITÉ
with onglet_e:
    st.write("Compare le comportement du modèle selon les groupes (variables sensibles). "
             "Une variable sensible peut influencer le modèle même indirectement.")
    nom_e = st.selectbox("Modèle", list(pipelines), key="eq_modele")
    seuil_e = st.slider("Seuil", 0.05, 0.95, seuil_global, 0.01, key="seuil_eq")
    sensibles = [c for c in ["SEX", "AGE", "MARRIAGE", "EDUCATION"] if c in X_test.columns]
    if not sensibles:
        st.info("Aucune variable sensible connue (SEX, AGE, MARRIAGE, EDUCATION) dans les variables retenues.")
    else:
        proba = st.session_state["probas"][nom_e]
        d = X_test[sensibles].copy()
        d["_reel"], d["_pred"] = y_test.values, (proba >= seuil_e).astype(int)
        if "AGE" in d:
            d["AGE (tranche)"] = pd.cut(d["AGE"], [0, 30, 40, 50, 60, 120])
        groupes = [c for c in sensibles if c != "AGE"] + (["AGE (tranche)"] if "AGE" in d else [])
        for col in groupes:
            t = d.groupby(col, observed=True).apply(lambda x: pd.Series({
                "Taux réel de la classe d'intérêt": x["_reel"].mean(),
                "Taux de clients signalés": x["_pred"].mean(),
                "Rappel": recall_score(x["_reel"], x["_pred"], zero_division=0),
                "Effectif": len(x)}), include_groups=False).round(3)
            st.markdown(f"**Par {col}**")
            st.dataframe(t, width="stretch")
