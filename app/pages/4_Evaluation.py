import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from core.metrics import metriques, tableau_comparatif

st.title("4. Évaluation")

if "pipelines" not in st.session_state:
    st.warning("Entraîne d'abord des modèles (page 3).")
    st.stop()

y_test, probas = st.session_state["y_test"], st.session_state["probas"]

seuil = st.slider("Seuil de décision", 0.05, 0.95, float(st.session_state.get("seuil", 0.5)), 0.01)
st.session_state["seuil"] = seuil

st.subheader(f"Comparaison des modèles au seuil {seuil:.2f}")
st.dataframe(tableau_comparatif(y_test, probas, seuil), width="stretch")
st.caption("Gini = 2·AUC − 1. KS = écart maximal entre les distributions des bons et mauvais payeurs.")

g1, g2 = st.columns(2)
with g1:
    fig = go.Figure()
    for nom, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{nom} ({roc_auc_score(y_test, p):.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="Hasard"))
    fig.update_layout(title="Courbes ROC", xaxis_title="Taux de faux positifs", yaxis_title="Rappel")
    st.plotly_chart(fig, width="stretch")
with g2:
    fig = go.Figure()
    for nom, p in probas.items():
        pr, rc, _ = precision_recall_curve(y_test, p)
        fig.add_trace(go.Scatter(x=rc, y=pr, mode="lines", name=f"{nom} ({average_precision_score(y_test, p):.3f})"))
    fig.add_hline(y=float(y_test.mean()), line_dash="dash", line_color="grey")
    fig.update_layout(title="Courbes précision-rappel", xaxis_title="Rappel", yaxis_title="Précision")
    st.plotly_chart(fig, width="stretch")

st.subheader("Détail d'un modèle")
nom = st.selectbox("Modèle", list(probas), index=0)
p = probas[nom]
pred = (p >= seuil).astype(int)
m = metriques(y_test, p, seuil)

a, b, c, d = st.columns(4)
a.metric("Précision", f"{m['Précision']:.3f}")
b.metric("Rappel", f"{m['Rappel']:.3f}")
c.metric("F1", f"{m['F1']:.3f}")
d.metric("Clients signalés", f"{pred.mean():.1%}")

yt = np.asarray(y_test)
matrice = np.array([[((yt == 0) & (pred == 0)).sum(), ((yt == 0) & (pred == 1)).sum()],
                    [((yt == 1) & (pred == 0)).sum(), ((yt == 1) & (pred == 1)).sum()]])
e1, e2 = st.columns(2)
with e1:
    st.plotly_chart(px.imshow(matrice, text_auto=True, color_continuous_scale="Blues",
                              x=["Prédit : non", "Prédit : oui"], y=["Réel : non", "Réel : oui"],
                              title=f"Matrice de confusion ({nom})"), width="stretch")
with e2:
    seuils = np.round(np.arange(0.05, 0.96, 0.05), 2)
    lignes = [metriques(y_test, p, s) for s in seuils]
    fig = go.Figure()
    for cle in ["Précision", "Rappel", "F1"]:
        fig.add_trace(go.Scatter(x=seuils, y=[l[cle] for l in lignes], mode="lines", name=cle))
    fig.add_vline(x=seuil, line_dash="dash")
    fig.update_layout(title="Effet du seuil", xaxis_title="Seuil")
    st.plotly_chart(fig, width="stretch")
