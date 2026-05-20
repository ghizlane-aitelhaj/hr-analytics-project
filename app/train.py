import os
import pandas as pd
import mlflow
import mlflow.sklearn
from snowflake.connector import connect
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("HR_Attrition_Prediction")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES ÉTENDUES — 35 colonnes au lieu de 18
# Ajouts clés : JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance,
# YearsAtCompany, YearsSinceLastPromotion, StockOptionLevel, NumCompaniesWorked,
# JobLevel, JobRole (9 modalités), EducationField (6 modalités)
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_FEATURES = [
    # ── Numériques ────────────────────────────────────────────────────────────
    "AGE",
    "DISTANCEFROMHOME",
    "EDUCATION",
    "MONTHLYINCOME",
    "TOTALWORKINGYEARS",
    "JOBLEVEL",                   # Nouveau — niveau hiérarchique
    "JOBSATISFACTION",            # Nouveau — satisfaction au travail (1-4)
    "ENVIRONMENTSATISFACTION",    # Nouveau — qualité environnement (1-4)
    "WORKLIFEBALANCE",            # Nouveau — équilibre vie pro/perso (1-4)
    "YEARSATCOMPANY",             # Nouveau — ancienneté
    "YEARSINCURRENTROLE",         # Nouveau — durée dans le rôle actuel
    "YEARSSINCELASTPROMOTION",    # Nouveau — temps depuis dernière promotion
    "YEARSWITHCURRMANAGER",       # Nouveau — stabilité managériale
    "NUMCOMPANIESWORKED",         # Nouveau — mobilité historique
    "STOCKOPTIONLEVEL",           # Nouveau — intéressement (0-3)
    "TRAININGTIMESLASTYEAR",      # Nouveau — formations reçues
    "JOBINVOLVEMENT",             # Nouveau — implication dans le travail (1-4)
    "RELATIONSHIPSATISFACTION",   # Nouveau — qualité des relations (1-4)

    # ── BusinessTravel ────────────────────────────────────────────────────────
    "BUSINESSTRAVEL_RARELY",
    "BUSINESSTRAVEL_FREQUENTLY",
    "BUSINESSTRAVEL_NONTRAVEL",

    # ── Department ───────────────────────────────────────────────────────────
    "DEPARTMENT_SALES",
    "DEPARTMENT_RND",
    "DEPARTMENT_HR",

    # ── EducationField (Nouveau) ──────────────────────────────────────────────
    "EDUCATIONFIELD_LIFESCIENCES",
    "EDUCATIONFIELD_MEDICAL",
    "EDUCATIONFIELD_MARKETING",
    "EDUCATIONFIELD_TECHNICAL",
    "EDUCATIONFIELD_OTHER",
    "EDUCATIONFIELD_HUMANRESOURCES",

    # ── Gender ───────────────────────────────────────────────────────────────
    "GENDER_MALE",
    "GENDER_FEMALE",

    # ── JobRole (Nouveau) ─────────────────────────────────────────────────────
    "JOBROLE_SALESEXECUTIVE",
    "JOBROLE_RESEARCHSCIENTIST",
    "JOBROLE_LABORATORYTECHNICIAN",
    "JOBROLE_MANAGER",
    "JOBROLE_HEALTHCAREREPRESENTATIVE",
    "JOBROLE_MANUFACTURINGDIRECTOR",
    "JOBROLE_SALESREPRESENTATIVE",
    "JOBROLE_RESEARCHDIRECTOR",
    "JOBROLE_HUMANRESOURCES",

    # ── MaritalStatus ─────────────────────────────────────────────────────────
    "MARITALSTATUS_SINGLE",
    "MARITALSTATUS_MARRIED",
    "MARITALSTATUS_DIVORCED",

    # ── OverTime ──────────────────────────────────────────────────────────────
    "OVERTIME_YES",
    "OVERTIME_NO",
]


def train_model():
    ctx = connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASS'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema='GOLD',
        role=os.getenv('SNOWFLAKE_ROLE'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE')
    )

    db = os.getenv('SNOWFLAKE_DATABASE', 'RH_DB')
    schema = os.getenv('SNOWFLAKE_SCHEMA', 'GOLD')
    query = f'SELECT * FROM "{db}"."{schema}"."HR_EMPLOYEE_ENCODED"'
    df = pd.read_sql(query, ctx)
    ctx.close()

    # Normaliser les noms de colonnes
    df.columns = [c.strip().upper() for c in df.columns]

    print(f"Colonnes disponibles dans Snowflake : {list(df.columns)}")

    # Vérifier quelles features sont disponibles
    available = [f for f in REQUIRED_FEATURES if f in df.columns]
    missing = [f for f in REQUIRED_FEATURES if f not in df.columns]
    if missing:
        print(f"\n⚠️  Features manquantes dans Snowflake (initialisées à 0) : {missing}")
        for col in missing:
            df[col] = 0

    df = df[available + [f for f in missing] + ['ATTRITION']]

    print(f"\nDataset shape: {df.shape}")
    print(f"Features utilisées : {len(REQUIRED_FEATURES)}")

    X = df[REQUIRED_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['ATTRITION'].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1'] else 0)

    print(f"Distribution target : {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "n_estimators":      [100, 200],
        "max_depth":         [3, 5, 7],
        "min_samples_split": [10, 20],
        "min_samples_leaf":  [5, 10],
        "max_features":      ["sqrt"],
    }

    base_model = RandomForestClassifier(random_state=42, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_model, param_grid=param_grid,
        scoring="f1", cv=cv, n_jobs=-1, verbose=1,
    )

    with mlflow.start_run():
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)

        fi = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n--- Feature Importance (top 15) ---")
        print(fi.head(15).to_string(index=False))

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("best_cv_f1", grid.best_score_)
        mlflow.log_metric("accuracy",   acc)
        mlflow.log_metric("f1_score",   f1)
        mlflow.log_metric("n_features", len(REQUIRED_FEATURES))

        mlflow.sklearn.log_model(
            best_model, "attrition_model",
            registered_model_name="RF_Attrition_v1",
        )

        client = MlflowClient()
        model_name = "RF_Attrition_v1"

        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            prod_versions = [v for v in versions if v.current_stage == "Production"]
            old_f1 = (
                client.get_run(prod_versions[0].run_id).data.metrics.get('f1_score', 0)
                if prod_versions else 0
            )
        except Exception:
            old_f1 = 0

        if f1 > old_f1:
            print(f"\n✓ F1 nouveau ({f1:.3f}) > F1 Production ({old_f1:.3f}) → promotion.")
            all_versions = client.search_model_versions(f"name='{model_name}'")
            latest = max(int(v.version) for v in all_versions)
            client.transition_model_version_stage(
                name=model_name, version=str(latest),
                stage="Production", archive_existing_versions=True,
            )
            print(f"  Version {latest} → Production ✓")
        else:
            print(f"\n✗ F1 nouveau ({f1:.3f}) ≤ F1 Production ({old_f1:.3f}). Ancien modèle conservé.")

        print("\n=== RÉSULTATS ===")
        print(f"Meilleurs paramètres : {grid.best_params_}")
        print(f"Accuracy test        : {acc:.4f}")
        print(f"F1 test              : {f1:.4f}")
        print(f"CV F1 moyen          : {grid.best_score_:.4f}")
        print(f"Nombre de features   : {len(REQUIRED_FEATURES)}")


if __name__ == "__main__":
    train_model()
    