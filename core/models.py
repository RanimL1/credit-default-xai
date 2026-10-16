from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

NOMS_MODELES = [
    "Régression logistique", "Arbre de décision", "k plus proches voisins",
    "Random Forest", "XGBoost",
]


def creer_modele(nom, params, ratio=1.0, poids=True, seed=42, n_jobs=2):
    cw = "balanced" if poids else None
    if nom == "Régression logistique":
        return LogisticRegression(max_iter=2000, class_weight=cw, random_state=seed, **params)
    if nom == "Arbre de décision":
        return DecisionTreeClassifier(class_weight=cw, random_state=seed, **params)
    if nom == "k plus proches voisins":
        return KNeighborsClassifier(**params)
    if nom == "Random Forest":
        return RandomForestClassifier(class_weight=cw, n_jobs=n_jobs, random_state=seed, **params)
    if nom == "XGBoost":
        return XGBClassifier(scale_pos_weight=ratio if poids else 1.0, eval_metric="logloss",
                             n_jobs=n_jobs, random_state=seed, **params)
    raise ValueError(nom)


# Grilles pour RandomizedSearchCV (préfixe clf__ = étape "clf" du Pipeline)
GRILLES = {
    "Régression logistique": {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]},
    "Arbre de décision": {"clf__max_depth": [3, 4, 5, 6, 8], "clf__min_samples_leaf": [10, 20, 50, 100]},
    "k plus proches voisins": {"clf__n_neighbors": [5, 15, 25, 50, 75], "clf__weights": ["uniform", "distance"]},
    "Random Forest": {
        "clf__n_estimators": [100, 200, 300], "clf__max_depth": [6, 8, 10, 14],
        "clf__min_samples_leaf": [1, 5, 10, 20], "clf__max_features": ["sqrt", 0.3, 0.5],
    },
    "XGBoost": {
        "clf__n_estimators": [100, 200, 300], "clf__max_depth": [3, 4, 5, 6],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1], "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__min_child_weight": [1, 3, 5],
    },
}