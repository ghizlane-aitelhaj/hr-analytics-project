import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import dice_ml

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_FEATURES = [
    "AGE",
    "DISTANCEFROMHOME",
    "EDUCATION",
    "MONTHLYINCOME",
    "TOTALWORKINGYEARS",
    "BUSINESSTRAVEL_RARELY",
    "BUSINESSTRAVEL_FREQUENTLY",
    "BUSINESSTRAVEL_NONTRAVEL",
    "DEPARTMENT_SALES",
    "DEPARTMENT_RND",
    "DEPARTMENT_HR",
    "GENDER_MALE",
    "GENDER_FEMALE",
    "MARITALSTATUS_SINGLE",
    "MARITALSTATUS_MARRIED",
    "MARITALSTATUS_DIVORCED",
    "OVERTIME_YES",
    "OVERTIME_NO",
]

# Features que DiCE est autorisé à modifier directement
# CORRECTION : On ne fait varier que OVERTIME_YES directement, OVERTIME_NO sera mis à jour par cohérence logique.
ACTIONABLE_FEATURES = [
    "MONTHLYINCOME",
    "OVERTIME_YES",
]

# Bornes réalistes pour les features continues actionnables
FEATURE_RANGES = {
    "MONTHLYINCOME": (1000, 25000),
    "OVERTIME_YES": (0, 1)
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset de référence pour DiCE (population synthétique représentative)
# ─────────────────────────────────────────────────────────────────────────────

def _build_reference_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200

    rows = []
    for _ in range(n):
        row = {f: 0 for f in REQUIRED_FEATURES}
        row["AGE"] = int(rng.integers(22, 60))
        row["DISTANCEFROMHOME"] = int(rng.integers(1, 50))
        row["EDUCATION"] = int(rng.integers(1, 5))
        row["MONTHLYINCOME"] = int(rng.integers(1500, 20000))
        row["TOTALWORKINGYEARS"] = int(rng.integers(0, 35))

        # One-hot catégoriels (exactement 1 colonne = 1 par groupe)
        travel = rng.choice(["BUSINESSTRAVEL_RARELY", "BUSINESSTRAVEL_FREQUENTLY", "BUSINESSTRAVEL_NONTRAVEL"])
        row[travel] = 1

        dept = rng.choice(["DEPARTMENT_SALES", "DEPARTMENT_RND", "DEPARTMENT_HR"])
        row[dept] = 1

        gender = rng.choice(["GENDER_MALE", "GENDER_FEMALE"])
        row[gender] = 1

        marital = rng.choice(["MARITALSTATUS_SINGLE", "MARITALSTATUS_MARRIED", "MARITALSTATUS_DIVORCED"])
        row[marital] = 1

        ot = rng.choice(["OVERTIME_YES", "OVERTIME_NO"])
        row[ot] = 1

        # CORRECTION : Pour DiCE, la cible d'entraînement doit correspondre au format dbt/Snowflake utilisé (0 ou 1)
        row["ATTRITION"] = int(rng.integers(0, 2))
        rows.append(row)

    return pd.DataFrame(rows)


REFERENCE_DF = _build_reference_dataset()

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    mlflow.set_tracking_uri("http://mlflow:5000")
    model_name = "RF_Attrition_v1"
    return mlflow.sklearn.load_model(f"models:/{model_name}/Production")


def _get_feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return REQUIRED_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# PRÉDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_risk(model, row_dict: dict) -> dict:
    feature_names = _get_feature_names(model)
    full_row = {fn: 0 for fn in feature_names}

    # Mapping catégoriels → colonnes one-hot
    mappings = {
        "GENDER": {"Male": "GENDER_MALE", "Female": "GENDER_FEMALE"},
        "MARITALSTATUS": {
            "Single": "MARITALSTATUS_SINGLE",
            "Married": "MARITALSTATUS_MARRIED",
            "Divorced": "MARITALSTATUS_DIVORCED",
        },
        "BUSINESSTRAVEL": {
            "Non-Travel": "BUSINESSTRAVEL_NONTRAVEL",
            "Travel_Rarely": "BUSINESSTRAVEL_RARELY",
            "Travel_Frequently": "BUSINESSTRAVEL_FREQUENTLY",
        },
        "DEPARTMENT": {
            "Sales": "DEPARTMENT_SALES",
            "Research & Development": "DEPARTMENT_RND",
            "Human Resources": "DEPARTMENT_HR",
        },
        "OVERTIME": {"No": "OVERTIME_NO", "Yes": "OVERTIME_YES"},
    }

    numeric_keys = {
        "AGE": "AGE",
        "DISTANCEFROMHOME": "DISTANCEFROMHOME",
        "EDUCATION": "EDUCATION",
        "MONTHLYINCOME": "MONTHLYINCOME",
        "TOTALWORKINGYEARS": "TOTALWORKINGYEARS",
    }

    for key, col in numeric_keys.items():
        if key in row_dict and row_dict[key] is not None:
            try:
                full_row[col] = float(row_dict[key])
            except (ValueError, TypeError):
                full_row[col] = 0.0

    for category, mapping in mappings.items():
        val = row_dict.get(category)
        if val in mapping:
            full_row[mapping[val]] = 1

    df = pd.DataFrame([full_row], columns=feature_names)

    proba = model.predict_proba(df)[0]
    risk_score = float(proba[1])
    prediction = "Risque Élevé" if risk_score > 0.5 else "Risque Faible"

    return {"risk_score": risk_score, "prediction": prediction, "df": df}


# ─────────────────────────────────────────────────────────────────────────────
# CONTREFACTUELS DiCE
# ─────────────────────────────────────────────────────────────────────────────

def get_counterfactuals(model, current_employee_df: pd.DataFrame, n_cfs: int = 3):
    try:
        # Préparation de l'historique synthétique combiné
        ref_with_current = pd.concat(
            [REFERENCE_DF.drop(columns=["ATTRITION"]), current_employee_df],
            ignore_index=True,
        )
        ref_with_label = ref_with_current.copy()
        
        # Label synthétique basé sur une règle métier cohérente
        ref_with_label["ATTRITION"] = (
            (ref_with_label["MONTHLYINCOME"] < 5000) | (ref_with_label["OVERTIME_YES"] == 1)
        ).astype(int)

        # Configuration de DiCE
        d = dice_ml.Data(
            dataframe=ref_with_label,
            continuous_features=["AGE", "DISTANCEFROMHOME", "MONTHLYINCOME", "TOTALWORKINGYEARS", "EDUCATION"],
            outcome_name="ATTRITION",
        )

        m = dice_ml.Model(model=model, backend="sklearn")
        exp = dice_ml.Dice(d, m, method="random")

        # CORRECTION : Suppression de OVERTIME_NO de la liste de variation directe.
        # DiCE fera varier OVERTIME_YES (0 ou 1) et notre parseur s'occupera d'ajuster OVERTIME_NO pour le modèle.
        dice_exp = exp.generate_counterfactuals(
            current_employee_df,
            total_CFs=n_cfs,
            desired_class=0,  # 0 = Rester dans l'entreprise
            features_to_vary=ACTIONABLE_FEATURES,
            permitted_range=FEATURE_RANGES,
        )

        return _parse_counterfactuals(dice_exp, current_employee_df, model)

    except Exception as e:
        print(f"[DiCE] Erreur génération contrefactuels : {e}")
        # En cas d'erreur de convergence de DiCE, l'UI récupère le plan B de manière transparente
        return _fallback_recommendations(current_employee_df, model)


def _parse_counterfactuals(dice_exp, original_df: pd.DataFrame, model) -> list[dict]:
    scenarios = []
    original_salary = float(original_df["MONTHLYINCOME"].iloc[0])
    original_ot_yes = int(original_df["OVERTIME_YES"].iloc[0])

    try:
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df
    except (IndexError, AttributeError):
        return []

    if cf_df is None or cf_df.empty:
        return []

    for _, cf_row in cf_df.iterrows():
        changes = []
        scenario = {
            "new_salary": None,
            "delta_salary": None,
            "remove_overtime": False,
            "new_risk_proba": None,
            "changes": [],
        }

        # Traitement du Salaire
        new_salary = float(cf_row.get("MONTHLYINCOME", original_salary))
        if abs(new_salary - original_salary) > 50:
            delta = new_salary - original_salary
            scenario["new_salary"] = round(new_salary, 0)
            scenario["delta_salary"] = round(delta, 0)
            pct = (delta / original_salary) * 100
            direction = "Augmenter" if delta > 0 else "Réduire"
            changes.append(f"{direction} le salaire à {new_salary:,.0f} € ({pct:+.1f}%)")

        # CORRECTION : Gestion propre de la contrainte One-Hot Encodée pour les heures supplémentaires
        # Si DiCE propose une valeur continue entre 0 et 1 (ex: 0.3), on applique un seuil à 0.5
        raw_ot_yes = cf_row.get("OVERTIME_YES", original_ot_yes)
        new_ot_yes = 1 if raw_ot_yes >= 0.5 else 0
        
        if original_ot_yes == 1 and new_ot_yes == 0:
            scenario["remove_overtime"] = True
            changes.append("Supprimer les heures supplémentaires obligatoires")
        elif original_ot_yes == 0 and new_ot_yes == 1:
            changes.append("Note : Activer les heures supplémentaires (scénario théorique)")

        if not changes:
            continue  # Pas de changement visible → on passe au scénario suivant

        # Re-calcul strict de la probabilité à l'aide du vrai modèle RandomForest
        cf_features = original_df.copy()
        cf_features["MONTHLYINCOME"] = new_salary
        
        # CORRECTION LOGIQUE : On garantit la somme stricte des colonnes binaires (One-Hot)
        cf_features["OVERTIME_YES"] = new_ot_yes
        cf_features["OVERTIME_NO"] = 1 - new_ot_yes

        try:
            # Récupération de la probabilité exacte ajustée
            new_proba = float(model.predict_proba(cf_features)[0][1])
        except Exception:
            new_proba = None

        scenario["new_risk_proba"] = new_proba
        scenario["changes"] = changes
        scenarios.append(scenario)

    return scenarios


def _fallback_recommendations(original_df: pd.DataFrame, model) -> list[dict]:
    salary = float(original_df["MONTHLYINCOME"].iloc[0])
    ot_yes = int(original_df["OVERTIME_YES"].iloc[0])
    scenarios = []

    # Option Heuristique 1 & 2 : Recommandation d'augmentations salariales directes
    for pct in [0.10, 0.20]:
        new_salary = round(salary * (1 + pct))
        cf = original_df.copy()
        cf["MONTHLYINCOME"] = new_salary
        try:
            new_proba = float(model.predict_proba(cf)[0][1])
        except Exception:
            new_proba = None
        scenarios.append({
            "new_salary": new_salary,
            "delta_salary": round(new_salary - salary),
            "remove_overtime": False,
            "new_risk_proba": new_proba,
            "changes": [f"Augmenter le salaire à {new_salary:,.0f} € (+{pct*100:.0f}%)"],
        })

    # Option Heuristique 3 : Suppression des Heures Supplémentaires
    if ot_yes == 1:
        cf = original_df.copy()
        cf["OVERTIME_YES"] = 0
        cf["OVERTIME_NO"] = 1
        try:
            new_proba = float(model.predict_proba(cf)[0][1])
        except Exception:
            new_proba = None
        scenarios.append({
            "new_salary": None,
            "delta_salary": None,
            "remove_overtime": True,
            "new_risk_proba": new_proba,
            "changes": ["Supprimer les heures supplémentaires obligatoires"],
        })

    return scenarios