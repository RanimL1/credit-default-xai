from pydantic import BaseModel, Field


class Client(BaseModel):
    """Un client de la banque (variables du jeu Default of Credit Card Clients)."""

    LIMIT_BAL: float = Field(..., ge=0, description="Limite de crédit (NT$)")
    SEX: int = Field(..., ge=1, le=2, description="1 = homme, 2 = femme")
    EDUCATION: int = Field(..., ge=1, le=4, description="1 = master/doctorat, 2 = université, 3 = lycée, 4 = autre")
    MARRIAGE: int = Field(..., ge=1, le=3, description="1 = marié, 2 = célibataire, 3 = autre")
    AGE: int = Field(..., ge=18, le=100, description="Âge en années")

    PAY_1: int = Field(..., ge=-2, le=9, description="Statut de remboursement, septembre (-2/-1/0 = à jour, 1..9 = mois de retard)")
    PAY_2: int = Field(..., ge=-2, le=9, description="Statut de remboursement, août")
    PAY_3: int = Field(..., ge=-2, le=9, description="Statut de remboursement, juillet")
    PAY_4: int = Field(..., ge=-2, le=9, description="Statut de remboursement, juin")
    PAY_5: int = Field(..., ge=-2, le=9, description="Statut de remboursement, mai")
    PAY_6: int = Field(..., ge=-2, le=9, description="Statut de remboursement, avril")

    BILL_AMT1: float = Field(..., description="Montant de la facture, septembre")
    BILL_AMT2: float = Field(..., description="Montant de la facture, août")
    BILL_AMT3: float = Field(..., description="Montant de la facture, juillet")
    BILL_AMT4: float = Field(..., description="Montant de la facture, juin")
    BILL_AMT5: float = Field(..., description="Montant de la facture, mai")
    BILL_AMT6: float = Field(..., description="Montant de la facture, avril")

    PAY_AMT1: float = Field(..., ge=0, description="Montant payé, septembre")
    PAY_AMT2: float = Field(..., ge=0, description="Montant payé, août")
    PAY_AMT3: float = Field(..., ge=0, description="Montant payé, juillet")
    PAY_AMT4: float = Field(..., ge=0, description="Montant payé, juin")
    PAY_AMT5: float = Field(..., ge=0, description="Montant payé, mai")
    PAY_AMT6: float = Field(..., ge=0, description="Montant payé, avril")

    model_config = {
        "json_schema_extra": {
            "example": {
                "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
                "PAY_1": 2, "PAY_2": 2, "PAY_3": 2, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 48000, "BILL_AMT2": 47000, "BILL_AMT3": 46000,
                "BILL_AMT4": 30000, "BILL_AMT5": 29000, "BILL_AMT6": 28000,
                "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 1500,
                "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 1000,
            }
        }
    }


class Requete(BaseModel):
    client: Client
    modele: str = Field("XGBoost", description="XGBoost, Random Forest, Régression logistique, Arbre de décision, k plus proches voisins")
    seuil: float = Field(0.5, ge=0, le=1, description="Seuil de décision sur la probabilité de défaut")


class Prediction(BaseModel):
    modele: str
    probabilite_defaut: float
    seuil: float
    decision: str = Field(..., description="'defaut_probable' si probabilité >= seuil, sinon 'pas_de_defaut'")