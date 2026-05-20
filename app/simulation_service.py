import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import dice_ml

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES ÉTENDUES — 47 colonnes
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_FEATURES = [
    # Numériques
    "AGE", "DISTANCEFROMHOME", "EDUCATION", "MONTHLYINCOME", "TOTALWORKINGYEARS",
    "JOBLEVEL", "JOBSATISFACTION", "ENVIRONMENTSATISFACTION", "WORKLIFEBALANCE",
    "YEARSATCOMPANY", "YEARSINCURRENTROLE", "YEARSSINCELASTPROMOTION",
    "YEARSWITHCURRMANAGER", "NUMCOMPANIESWORKED", "STOCKOPTIONLEVEL",
    "TRAININGTIMESLASTYEAR", "JOBINVOLVEMENT", "RELATIONSHIPSATISFACTION",
    # BusinessTravel
    "BUSINESSTRAVEL_RARELY", "BUSINESSTRAVEL_FREQUENTLY", "BUSINESSTRAVEL_NONTRAVEL",
    # Department
    "DEPARTMENT_SALES", "DEPARTMENT_RND", "DEPARTMENT_HR",
    # EducationField
    "EDUCATIONFIELD_LIFESCIENCES", "EDUCATIONFIELD_MEDICAL", "EDUCATIONFIELD_MARKETING",
    "EDUCATIONFIELD_TECHNICAL", "EDUCATIONFIELD_OTHER", "EDUCATIONFIELD_HUMANRESOURCES",
    # Gender
    "GENDER_MALE", "GENDER_FEMALE",
    # JobRole
    "JOBROLE_SALESEXECUTIVE", "JOBROLE_RESEARCHSCIENTIST", "JOBROLE_LABORATORYTECHNICIAN",
    "JOBROLE_MANAGER", "JOBROLE_HEALTHCAREREPRESENTATIVE", "JOBROLE_MANUFACTURINGDIRECTOR",
    "JOBROLE_SALESREPRESENTATIVE", "JOBROLE_RESEARCHDIRECTOR", "JOBROLE_HUMANRESOURCES",
    # MaritalStatus
    "MARITALSTATUS_SINGLE", "MARITALSTATUS_MARRIED", "MARITALSTATUS_DIVORCED",
    # OverTime
    "OVERTIME_YES", "OVERTIME_NO",
]

# Features actionnables par le manager → DiCE peut les modifier
ACTIONABLE_FEATURES = ["MONTHLYINCOME"]

FEATURE_RANGES = {
    "MONTHLYINCOME": (1000, 25000),
}

CONTINUOUS_FEATURES = [
    "AGE", "DISTANCEFROMHOME", "MONTHLYINCOME", "TOTALWORKINGYEARS",
    "YEARSATCOMPANY", "YEARSINCURRENTROLE", "YEARSSINCELASTPROMOTION",
    "YEARSWITHCURRMANAGER", "NUMCOMPANIESWORKED", "STOCKOPTIONLEVEL",
    "TRAININGTIMESLASTYEAR", "JOBSATISFACTION", "ENVIRONMENTSATISFACTION",
    "WORKLIFEBALANCE", "JOBINVOLVEMENT", "RELATIONSHIPSATISFACTION",
    "JOBLEVEL", "EDUCATION",
]


def _build_reference_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 300
    rows = []
    for _ in range(n):
        row = {f: 0 for f in REQUIRED_FEATURES}

        # Numériques
        row["AGE"] = int(rng.integers(22, 60))
        row["DISTANCEFROMHOME"] = int(rng.integers(1, 50))
        row["EDUCATION"] = int(rng.integers(1, 5))
        row["MONTHLYINCOME"] = int(rng.integers(1500, 20000))
        row["TOTALWORKINGYEARS"] = int(rng.integers(0, 35))
        row["JOBLEVEL"] = int(rng.integers(1, 5))
        row["JOBSATISFACTION"] = int(rng.integers(1, 4))
        row["ENVIRONMENTSATISFACTION"] = int(rng.integers(1, 4))
        row["WORKLIFEBALANCE"] = int(rng.integers(1, 4))
        row["YEARSATCOMPANY"] = int(rng.integers(0, 25))
        row["YEARSINCURRENTROLE"] = int(rng.integers(0, 15))
        row["YEARSSINCELASTPROMOTION"] = int(rng.integers(0, 15))
        row["YEARSWITHCURRMANAGER"] = int(rng.integers(0, 17))
        row["NUMCOMPANIESWORKED"] = int(rng.integers(0, 9))
        row["STOCKOPTIONLEVEL"] = int(rng.integers(0, 3))
        row["TRAININGTIMESLASTYEAR"] = int(rng.integers(0, 6))
        row["JOBINVOLVEMENT"] = int(rng.integers(1, 4))
        row["RELATIONSHIPSATISFACTION"] = int(rng.integers(1, 4))

        # One-hot catégoriels
        for group, cols in [
            ("travel", ["BUSINESSTRAVEL_RARELY", "BUSINESSTRAVEL_FREQUENTLY", "BUSINESSTRAVEL_NONTRAVEL"]),
            ("dept",   ["DEPARTMENT_SALES", "DEPARTMENT_RND", "DEPARTMENT_HR"]),
            ("edu",    ["EDUCATIONFIELD_LIFESCIENCES", "EDUCATIONFIELD_MEDICAL", "EDUCATIONFIELD_MARKETING",
                        "EDUCATIONFIELD_TECHNICAL", "EDUCATIONFIELD_OTHER", "EDUCATIONFIELD_HUMANRESOURCES"]),
            ("gender", ["GENDER_MALE", "GENDER_FEMALE"]),
            ("role",   ["JOBROLE_SALESEXECUTIVE", "JOBROLE_RESEARCHSCIENTIST", "JOBROLE_LABORATORYTECHNICIAN",
                        "JOBROLE_MANAGER", "JOBROLE_HEALTHCAREREPRESENTATIVE", "JOBROLE_MANUFACTURINGDIRECTOR",
                        "JOBROLE_SALESREPRESENTATIVE", "JOBROLE_RESEARCHDIRECTOR", "JOBROLE_HUMANRESOURCES"]),
            ("marital",["MARITALSTATUS_SINGLE", "MARITALSTATUS_MARRIED", "MARITALSTATUS_DIVORCED"]),
            ("ot",     ["OVERTIME_YES", "OVERTIME_NO"]),
        ]:
            chosen = rng.choice(cols)
            row[chosen] = 1

        row["ATTRITION"] = int(rng.integers(0, 2))
        rows.append(row)

    return pd.DataFrame(rows)


REFERENCE_DF = _build_reference_dataset()


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    return mlflow.sklearn.load_model("models:/RF_Attrition_v1/Production")


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

    # Mappings catégoriels
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
        "EDUCATIONFIELD": {
            "Life Sciences": "EDUCATIONFIELD_LIFESCIENCES",
            "Medical": "EDUCATIONFIELD_MEDICAL",
            "Marketing": "EDUCATIONFIELD_MARKETING",
            "Technical Degree": "EDUCATIONFIELD_TECHNICAL",
            "Other": "EDUCATIONFIELD_OTHER",
            "Human Resources": "EDUCATIONFIELD_HUMANRESOURCES",
        },
        "JOBROLE": {
            "Sales Executive": "JOBROLE_SALESEXECUTIVE",
            "Research Scientist": "JOBROLE_RESEARCHSCIENTIST",
            "Laboratory Technician": "JOBROLE_LABORATORYTECHNICIAN",
            "Manager": "JOBROLE_MANAGER",
            "Healthcare Representative": "JOBROLE_HEALTHCAREREPRESENTATIVE",
            "Manufacturing Director": "JOBROLE_MANUFACTURINGDIRECTOR",
            "Sales Representative": "JOBROLE_SALESREPRESENTATIVE",
            "Research Director": "JOBROLE_RESEARCHDIRECTOR",
            "Human Resources": "JOBROLE_HUMANRESOURCES",
        },
    }

    numeric_keys = {
        "AGE": "AGE", "DISTANCEFROMHOME": "DISTANCEFROMHOME",
        "EDUCATION": "EDUCATION", "MONTHLYINCOME": "MONTHLYINCOME",
        "TOTALWORKINGYEARS": "TOTALWORKINGYEARS", "JOBLEVEL": "JOBLEVEL",
        "JOBSATISFACTION": "JOBSATISFACTION",
        "ENVIRONMENTSATISFACTION": "ENVIRONMENTSATISFACTION",
        "WORKLIFEBALANCE": "WORKLIFEBALANCE", "YEARSATCOMPANY": "YEARSATCOMPANY",
        "YEARSINCURRENTROLE": "YEARSINCURRENTROLE",
        "YEARSSINCELASTPROMOTION": "YEARSSINCELASTPROMOTION",
        "YEARSWITHCURRMANAGER": "YEARSWITHCURRMANAGER",
        "NUMCOMPANIESWORKED": "NUMCOMPANIESWORKED",
        "STOCKOPTIONLEVEL": "STOCKOPTIONLEVEL",
        "TRAININGTIMESLASTYEAR": "TRAININGTIMESLASTYEAR",
        "JOBINVOLVEMENT": "JOBINVOLVEMENT",
        "RELATIONSHIPSATISFACTION": "RELATIONSHIPSATISFACTION",
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

    # Ne garder que les colonnes connues du modèle
    df = pd.DataFrame([{k: full_row.get(k, 0) for k in feature_names}], columns=feature_names)

    proba = model.predict_proba(df)[0]
    risk_score = float(proba[1])
    prediction = "Risque Élevé" if risk_score > 0.5 else "Risque Faible"

    return {"risk_score": risk_score, "prediction": prediction, "df": df}


# ─────────────────────────────────────────────────────────────────────────────
# CONTREFACTUELS DiCE
# ─────────────────────────────────────────────────────────────────────────────

def get_counterfactuals(model, current_employee_df: pd.DataFrame, n_cfs: int = 3) -> list[dict]:
    try:
        ref = pd.concat(
            [REFERENCE_DF.drop(columns=["ATTRITION"]), current_employee_df],
            ignore_index=True,
        )
        ref_labeled = ref.copy()
        ref_labeled["ATTRITION"] = (
            (ref_labeled["MONTHLYINCOME"] < 5000) | (ref_labeled["OVERTIME_YES"] == 1)
        ).astype(int)

        # Utiliser uniquement les features connues du modèle
        feature_names = _get_feature_names(model)
        continuous = [f for f in CONTINUOUS_FEATURES if f in feature_names]

        d = dice_ml.Data(
            dataframe=ref_labeled[[f for f in feature_names if f in ref_labeled.columns] + ["ATTRITION"]],
            continuous_features=continuous,
            outcome_name="ATTRITION",
        )
        m = dice_ml.Model(model=model, backend="sklearn")
        exp = dice_ml.Dice(d, m, method="random")

        cf_df_input = current_employee_df[[f for f in feature_names if f in current_employee_df.columns]]

        dice_exp = exp.generate_counterfactuals(
            cf_df_input,
            total_CFs=n_cfs,
            desired_class=0,
            features_to_vary=["MONTHLYINCOME"],
            permitted_range=FEATURE_RANGES,
        )

        return _parse_counterfactuals(dice_exp, current_employee_df, model)

    except Exception as e:
        print(f"[DiCE] Erreur : {e}")
        return _fallback_recommendations(current_employee_df, model)


def _parse_counterfactuals(dice_exp, original_df, model) -> list[dict]:
    scenarios = []
    original_salary = float(original_df["MONTHLYINCOME"].iloc[0])

    try:
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df
    except (IndexError, AttributeError):
        return []

    for _, cf_row in cf_df.iterrows():
        changes = []
        new_salary = float(cf_row.get("MONTHLYINCOME", original_salary))
        if abs(new_salary - original_salary) > 50:
            delta = new_salary - original_salary
            pct = (delta / original_salary) * 100
            direction = "Augmenter" if delta > 0 else "Réduire"
            changes.append(f"{direction} le salaire à {new_salary:,.0f} € ({pct:+.1f}%)")

        if not changes:
            continue

        cf_features = original_df.copy()
        cf_features["MONTHLYINCOME"] = new_salary
        try:
            new_proba = float(model.predict_proba(cf_features)[0][1])
        except Exception:
            new_proba = None

        scenarios.append({
            "new_salary": round(new_salary),
            "delta_salary": round(new_salary - original_salary),
            "remove_overtime": False,
            "new_risk_proba": new_proba,
            "changes": changes,
        })

    return scenarios


def _fallback_recommendations(original_df, model) -> list[dict]:
    """
    Si DiCE échoue, recommandations heuristiques enrichies avec les nouvelles features.
    """
    salary = float(original_df["MONTHLYINCOME"].iloc[0])
    ot_yes = int(original_df.get("OVERTIME_YES", pd.Series([0])).iloc[0])
    job_sat = float(original_df.get("JOBSATISFACTION", pd.Series([2])).iloc[0])
    promo = float(original_df.get("YEARSSINCELASTPROMOTION", pd.Series([0])).iloc[0])
    stock = float(original_df.get("STOCKOPTIONLEVEL", pd.Series([0])).iloc[0])
    training = float(original_df.get("TRAININGTIMESLASTYEAR", pd.Series([2])).iloc[0])

    scenarios = []

    # Recommandation salaire +15%
    new_salary = round(salary * 1.15)
    cf = original_df.copy()
    cf["MONTHLYINCOME"] = new_salary
    try:
        new_proba = float(model.predict_proba(cf)[0][1])
    except Exception:
        new_proba = None
    scenarios.append({
        "new_salary": new_salary, "delta_salary": round(new_salary - salary),
        "remove_overtime": False, "new_risk_proba": new_proba,
        "changes": [f"Augmenter le salaire à {new_salary:,.0f} € (+15%)"],
    })

    # Recommandation suppression heures sup
    if ot_yes == 1:
        cf2 = original_df.copy()
        cf2["OVERTIME_YES"] = 0
        cf2["OVERTIME_NO"] = 1
        try:
            new_proba2 = float(model.predict_proba(cf2)[0][1])
        except Exception:
            new_proba2 = None
        scenarios.append({
            "new_salary": None, "delta_salary": None,
            "remove_overtime": True, "new_risk_proba": new_proba2,
            "changes": ["Supprimer les heures supplémentaires obligatoires"],
        })

    # Recommandation satisfaction — si JobSatisfaction faible
    if job_sat <= 2:
        scenarios.append({
            "new_salary": None, "delta_salary": None,
            "remove_overtime": False, "new_risk_proba": None,
            "changes": ["Mener un entretien individuel pour identifier les sources d'insatisfaction au travail"],
        })

    # Recommandation promotion — si pas de promotion depuis > 3 ans
    if promo >= 3:
        scenarios.append({
            "new_salary": None, "delta_salary": None,
            "remove_overtime": False, "new_risk_proba": None,
            "changes": [f"Envisager une promotion ou revalorisation de poste ({promo:.0f} ans sans promotion)"],
        })

    # Recommandation stock-options — si niveau 0
    if stock == 0:
        scenarios.append({
            "new_salary": None, "delta_salary": None,
            "remove_overtime": False, "new_risk_proba": None,
            "changes": ["Attribuer des stock-options niveau 1 pour renforcer l'attachement à long terme"],
        })

    # Recommandation formation — si peu de formations
    if training <= 1:
        scenarios.append({
            "new_salary": None, "delta_salary": None,
            "remove_overtime": False, "new_risk_proba": None,
            "changes": ["Proposer un plan de formation : 2-3 sessions sur l'année pour développer les compétences"],
        })

    return scenarios