import pandas as pd

NOMS_TAIWAN = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)


def charger_taiwan():
    """Télécharge Default of Credit Card Clients (UCI, id=350) et applique le nettoyage de l'étape 1."""
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=350)
    X = ds.data.features.copy()
    y = ds.data.targets
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = y.rename("DEFAULT")
    if X.columns[0].startswith("X"):
        X.columns = NOMS_TAIWAN
    else:
        X = X.rename(columns={"PAY_0": "PAY_1"})
    df = pd.concat([X, y], axis=1)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df.drop_duplicates().reset_index(drop=True)


def charger_csv(fichier):
    return pd.read_csv(fichier)
