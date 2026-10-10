# Prédiction du défaut de paiement : application de classification et API

Projet **Challenge 1** du cours de Data Mining (ENSI, II3-IA, 2026-2027, enseignante : Rym Besrour).
Cas métier choisi : **scoring de crédit** (cas n°1 du catalogue) sur le jeu *Default of Credit Card Clients* (Taiwan, UCI).

L'objectif est de prédire si un client de carte bancaire fera défaut le mois suivant, de comparer plusieurs modèles de classification sur des métriques adaptées au risque de crédit, et de choisir un seuil de décision orienté vers la détection des défauts.


## Données

- Source : jeu *Default of Credit Card Clients* de l'UCI (identifiant 350), téléchargé automatiquement avec `ucimlrepo`. L'import d'un CSV personnel est aussi possible dans l'application.
- 29 965 clients après suppression des doublons, 23 variables explicatives (limite de crédit, sexe, niveau d'études, situation familiale, âge, statuts de remboursement, montants de factures et de paiements sur six mois).
- Cible : défaut de paiement le mois suivant. Environ **22 %** de défauts : les classes sont déséquilibrées.
- Nettoyage : regroupement des codes non documentés de `EDUCATION` (0, 5, 6 vers « autre ») et de `MARRIAGE` (0 vers « autre »), suppression des doublons, renommage de `PAY_0` en `PAY_1`.

## Choix méthodologiques

- **Split stratifié** 80 / 20 (graine 42) : le test garde la proportion réelle de défauts, il n'est jamais rééquilibré.
- **Pas de fuite de données** : imputation, encodage et normalisation sont dans un `Pipeline` scikit-learn, ajustés sur le train uniquement.
- **Déséquilibre des classes** traité par des poids de classes (`class_weight`, `scale_pos_weight`), sans sur-échantillonnage.
- **Optimisation** des hyperparamètres par `RandomizedSearchCV`, validation croisée stratifiée sur le train, critère AUC.
- **Métriques** : AUC-ROC, PR-AUC, précision, rappel, F1 sur la classe « défaut », plus l'indice de Gini et la statistique de Kolmogorov-Smirnov (spécifiques au scoring de crédit).
- **Seuil de décision** : un défaut non détecté coûte plus cher qu'un bon client refusé à tort. Le projet privilégie donc le **rappel** sous contrainte d'une précision minimale. Ce choix repose sur cette hypothèse de coûts, que le curseur de seuil de l'application permet de faire varier.

## Résultats

Modèles optimisés, évalués sur le jeu de test (seuil 0,5) :

| Modèle | AUC-ROC | PR-AUC |
|---|---|---|
| XGBoost | 0,775 | 0,557 |
| Random Forest | 0,774 | 0,553 |
| Arbre de décision | 0,754 | 0,510 |
| k plus proches voisins | 0,748 | 0,511 |
| Régression logistique | 0,719 | 0,503 |

Avec XGBoost au seuil 0,5, le rappel est d'environ 0,61 et la précision d'environ 0,47. Baisser le seuil détecte davantage de défauts au prix de plus de fausses alertes. Les modèles les plus performants sont les méthodes d'ensemble, mais l'écart avec la régression logistique (interprétable) reste modeste, ce qui nourrit la discussion performance contre interprétabilité.

## Structure du dépôt
 
```
api/                   API FastAPI (main.py, explain.py, schemas.py, requirements.txt)
app/                   Application Streamlit (Home.py et pages/)
core/                  Logique partagée (données, préparation, modèles, métriques, XAI, client de l'API)
models/                Modèles entraînés et jeu de test (joblib) utilisés par l'API
notebooks/             Notebook(s) d'analyse (explicabilité)
.streamlit/            Modèle de configuration (secrets.toml.example)
requirements.txt       Dépendances de l'application Streamlit
Dockerfile             Image de l'API (déploiement Hugging Face Spaces)
test_api.py            Test rapide de l'API
```

## Installation et lancement

Prérequis : Python 3.10 ou plus récent.

```bash
git clone https://github.com/RanimL1/credit-default-xai.git
cd credit-default-xai
python -m venv .venv
# Windows : .venv\Scripts\Activate.ps1      Linux/macOS : source .venv/bin/activate
```

### Application Streamlit

```bash
pip install -r requirements.txt
python -m streamlit run app/Home.py
```

L'application s'ouvre sur http://localhost:8501. Parcours : Données, Préparation, Modèles, Évaluation.

### API FastAPI

```bash
pip install -r requirements-api.txt
python -m uvicorn api.main:app --reload --port 8000
```

La documentation interactive est sur http://localhost:8000/docs. Pour tester : `python test_api.py`.

| Route | Rôle |
|---|---|
| `GET /health` | Vérifie que le service tourne |
| `GET /models` | Liste les modèles et les variables attendues |
| `POST /predict` | Probabilité de défaut et décision au seuil choisi |
| `POST /explain/shap` | Contributions SHAP d'un client et phrase de synthèse |
| `POST /explain/lime` | Contributions LIME d'un client |
| `POST /explain` | Tout en un : prédiction, SHAP, LIME, variables communes et phrase de synthèse |

Les modèles de `models/` ont été sauvegardés avec scikit-learn 1.6.1 et XGBoost 3.2.0 : `requirements-api.txt` fixe ces versions. Charger les modèles avec d'autres versions peut échouer ou donner des résultats différents.

## Limites
 
- Le jeu de données est ancien (Taïwan, 2005) et propre à un contexte bancaire : les résultats ne se transposent pas tels quels à un autre établissement.
- Le plafond de performance sur ce jeu est autour de 0,78 d'AUC dans la littérature : les gains supplémentaires viennent surtout de l'ingénierie de variables et du choix du seuil.
- L'explication d'un client par l'API concerne le jeu Taiwan et ses modèles pré-entraînés ; avec un autre jeu importé, seules les explications globales sont disponibles.
- SHAP n'est pas proposé pour le kNN (trop lent). Les valeurs SHAP se répartissent entre variables corrélées, et LIME dépend de son échantillonnage : les explications sont à lire comme des éclairages, pas comme des causes.
- Les variables sensibles (sexe, âge, situation familiale) font partie des données : leur usage dans un vrai système de crédit soulève des questions d'équité et de réglementation.
## Référence
 
Yeh, I-C., & Lien, C-H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2).
 
## Auteur
 
Ranim Latrous, étudiante en 3e année, spécialité intelligence artificielle, ENSI.