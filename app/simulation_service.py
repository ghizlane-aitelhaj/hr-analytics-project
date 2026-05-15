import mlflow
import mlflow.sklearn
import pandas as pd
#ma_modification
import dice_ml
# Liste des features réelles (one-hot encodées)
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

def load_model():
    mlflow.set_tracking_uri("http://mlflow:5000")
    model_name = "RF_Attrition_v1"
    return mlflow.sklearn.load_model(f"models:/{model_name}/Production")

def _get_feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return REQUIRED_FEATURES

def predict_risk(model, row_dict):
    """
    Prédire le risque d'attrition pour un candidat.
    row_dict contient les valeurs du formulaire dashboard
    """
    model = load_model()  # ← Recharger ici
    
    feature_names = _get_feature_names(model)
    
    # Initialiser toutes les colonnes à 0
    full_row = {fn: 0 for fn in feature_names}

    # Mapping des valeurs du formulaire aux colonnes du modèle
    mappings = {
        "GENDER": {
            "Male": "GENDER_MALE",
            "Female": "GENDER_FEMALE",
        },
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
        "OVERTIME": {
            "No": "OVERTIME_NO",
            "Yes": "OVERTIME_YES",
        },
    }

    # Remplir les valeurs numériques
    numeric_mappings = {
        "AGE": "AGE",
        "DISTANCEFROMHOME": "DISTANCEFROMHOME",
        "EDUCATION": "EDUCATION",
        "MONTHLYINCOME": "MONTHLYINCOME",
        "TOTALWORKINGYEARS": "TOTALWORKINGYEARS",
    }

    for key, col in numeric_mappings.items():
        if key in row_dict and row_dict[key] is not None:
            try:
                full_row[col] = float(row_dict[key])
            except:
                full_row[col] = 0

    # Remplir les valeurs catégoriques (one-hot encoding)
    for category, mapping in mappings.items():
        if category in row_dict and row_dict[category] in mapping:
            col_name = mapping[row_dict[category]]
            full_row[col_name] = 1

    # Créer DataFrame dans l'ordre correct
    df = pd.DataFrame([full_row], columns=feature_names)

    # Prédiction
    proba = model.predict_proba(df)[0]
    risk_score = float(proba[1])
    prediction = "Risque Élevé" if risk_score > 0.5 else "Risque Faible"
    
    return {"risk_score": risk_score, "prediction": prediction, "df": df}

#ma_modification
import dice_ml

def get_counterfactuals(model, current_employee_df):
    """
    Génère des alternatives pour transformer un risque "Élevé" en "Faible".
    """
    # 1. Préparation des données pour DiCE
    # Note: DiCE a besoin de connaître les types de colonnes.
    # On crée un dataset fictif basé sur tes REQUIRED_FEATURES pour l'initialisation
    d = dice_ml.Data(
        dataframe=current_employee_df.assign(ATTRITION=[1]), # Label temporaire
        continuous_features=['AGE', 'DISTANCEFROMHOME', 'MONTHLYINCOME', 'TOTALWORKINGYEARS', 'EDUCATION'],
        outcome_name='ATTRITION'
    )
    
    # 2. Configuration du modèle
    m = dice_ml.Model(model=model, backend="sklearn")
    
    # 3. Initialisation de l'expliquateur (Random est plus rapide pour le One-Hot)
    exp = dice_ml.Dice(d, m, method="random")
    
    # 4. Génération des scénarios
    # On interdit de changer l'âge, le genre, la distance ou l'expérience (variables non-actionnables)
    # On permet de changer : Salaire et Heures Sup (OVERTIME)
    try:
        dice_exp = exp.generate_counterfactuals(
            current_employee_df, 
            total_CFs=2, 
            desired_class=0, # On veut que la prédiction devienne "Reste"
            features_to_vary=['MONTHLYINCOME', 'OVERTIME_YES', 'OVERTIME_NO']
        )
        return dice_exp
    except Exception as e:
        print(f"Erreur DiCE: {e}")
        return None