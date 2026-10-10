import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def noms_variables(pipe):
    return [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]


def _dense(M):
    return M.toarray() if hasattr(M, "toarray") else np.asarray(M)


def transformer(pipe, X):
    return _dense(pipe.named_steps["prep"].transform(X))


def importance_native(pipe):
    clf, noms = pipe.named_steps["clf"], noms_variables(pipe)
    if hasattr(clf, "coef_"):
        return pd.Series(clf.coef_[0], index=noms)
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=noms)
    return None


def importance_permutation(pipe, X_test, y_test, n_repeats=3, seed=42, max_lignes=2000):
    if len(X_test) > max_lignes:
        X_test = X_test.sample(max_lignes, random_state=seed)
        y_test = y_test.loc[X_test.index]
    r = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                               n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    return pd.Series(r.importances_mean, index=X_test.columns)


def calcul_shap(pipe, X_train, X_echantillon, n_fond=500, seed=42):
    clf = pipe.named_steps["clf"]
    if isinstance(clf, KNeighborsClassifier):
        raise ValueError("SHAP n'est pas proposé pour le kNN (trop lent).")
    noms = noms_variables(pipe)
    Xs = transformer(pipe, X_echantillon)
    if isinstance(clf, LogisticRegression):
        fond = transformer(pipe, X_train.sample(min(n_fond, len(X_train)), random_state=seed))
        ex = shap.LinearExplainer(clf, fond)(Xs)
    else:
        ex = shap.TreeExplainer(clf)(Xs)
    vals, base = ex.values, np.asarray(ex.base_values)
    if vals.ndim == 3:                       # forêt / arbre : classe « positive »
        vals, base = vals[:, :, 1], base[:, 1]
    return shap.Explanation(values=vals, base_values=base, data=Xs, feature_names=noms)
