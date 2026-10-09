from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def construire_preprocesseur(colonnes_cat, colonnes_num, imputation="médiane", normalisation="Standard"):
    """Imputation + encodage one-hot + normalisation (ajustés sur le train uniquement, via le Pipeline)."""
    etapes_num = [("imp", SimpleImputer(strategy={"médiane": "median", "moyenne": "mean"}[imputation]))]
    if normalisation == "Standard":
        etapes_num.append(("sc", StandardScaler()))
    elif normalisation == "MinMax":
        etapes_num.append(("sc", MinMaxScaler()))

    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])
    transformateurs = []
    if colonnes_cat:
        transformateurs.append(("cat", cat, list(colonnes_cat)))
    if colonnes_num:
        transformateurs.append(("num", Pipeline(etapes_num), list(colonnes_num)))
    return ColumnTransformer(transformateurs)
