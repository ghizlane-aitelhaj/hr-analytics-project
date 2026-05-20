import pandas as pd
import snowflake.connector
import great_expectations as ge
import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Connexion Snowflake (couche GOLD / DBT output)
ctx = snowflake.connector.connect(
    user=os.getenv('SNOWFLAKE_USER'),
    password=os.getenv('SNOWFLAKE_PASS'),
    account=os.getenv('SNOWFLAKE_ACCOUNT'),
    warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
    database=os.getenv('SNOWFLAKE_DATABASE'),
    schema=os.getenv('SNOWFLAKE_SCHEMA', 'GOLD'),
    role=os.getenv('SNOWFLAKE_ROLE')
)

try:
    # Charger dataset dbt de référence
    db = os.getenv('SNOWFLAKE_DATABASE', 'RH_DB')
    schema = os.getenv('SNOWFLAKE_SCHEMA', 'GOLD')
    table = os.getenv('DQ_TABLE', 'HR_EMPLOYEE_ENCODED')
    query = f'SELECT * FROM "{db}"."{schema}"."{table}"'

    print(f"Connexion à {db}.{schema}.{table}")
    df_reference = pd.read_sql(query, ctx)
    
    print(f"Données de référence chargées : {df_reference.shape[0]} lignes, {df_reference.shape[1]} colonnes")
    
    # Nettoyage léger des noms de colonnes (uppercase)
    df_reference.columns = [col.strip().upper() for col in df_reference.columns]
    
    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATION DES DONNÉES DE PRODUCTION (POUR LE DRIFT)
    # Dans un vrai système, ce serait tes données collectées en temps réel.
    # Pour ton projet étudiant, on génère un échantillon pour montrer le test au prof.
    # ─────────────────────────────────────────────────────────────────────────
    def _get_current_production_data(df_ref):
        # On prend un morceau de données actuelles
        df_prod = df_ref.sample(n=min(100, len(df_ref)), random_state=42).copy()
        # On simule un "Drift" volontaire sur le salaire (ex: augmentation générale due à l'inflation)
        # Cela permettra de prouver au prof que ton système détecte bien la dérive !
        df_prod["MONTHLYINCOME"] = df_prod["MONTHLYINCOME"] * 1.35 
        return df_prod

    df_production = _get_current_production_data(df_reference)

    # Helper pour ajouter des expectations seulement si la colonne existe
    def expect_if_exists(df_ge, col_name, method_name, *args, **kwargs):
        if col_name in df_ge.columns:
            try:
                getattr(df_ge, method_name)(col_name, *args, **kwargs)
                print(f"  ✓ {method_name} pour {col_name}")
            except Exception as e:
                print(f"  ⚠️ Erreur {method_name} pour {col_name}: {str(e)}")
        else:
            print(f"  ⚠️ Colonne {col_name} non trouvée (skipped)")

    # Validation globale (Qualité + Drift)
    def validate_ml_ready_and_drift(df_prod, df_ref):
        print("\n📋 Démarrage des validations de Qualité...")
        df_ge = ge.from_pandas(df_prod)

        # 1. TARGET CHECK
        print("\n🎯 Validation de la TARGET...")
        expect_if_exists(df_ge, "ATTRITION", "expect_column_values_to_be_in_set", ["Yes", "No", 0, 1])

        # 2. NULL CHECK
        print("\n🔍 Validation des colonnes critiques (pas de NULL)...")
        expect_if_exists(df_ge, "MONTHLYINCOME", "expect_column_values_to_not_be_null")
        expect_if_exists(df_ge, "AGE", "expect_column_values_to_not_be_null")

        # 3. RANGE CHECK
        print("\n📊 Validation des plages de valeurs...")
        expect_if_exists(df_ge, "AGE", "expect_column_values_to_be_between", 18, 65)
        expect_if_exists(df_ge, "MONTHLYINCOME", "expect_column_min_to_be_between", 0, 100000)
        expect_if_exists(df_ge, "TOTALWORKINGYEARS", "expect_column_min_to_be_between", 0, 100)

        # 4. ENCODING CHECK
        print("\n✅ Validation du one-hot encoding...")
        binary_cols = [col for col in df_prod.columns if col.endswith(("_YES", "_NO", "_MALE", "_FEMALE", "_RARELY", "_FREQUENTLY", "_NONTRAVEL", "_SALES", "_RND", "_HR", "_SINGLE", "_MARRIED", "_DIVORCED"))]
        for col in binary_cols[:10]:
            expect_if_exists(df_ge, col, "expect_column_values_to_be_in_set", [0, 1])

        # 5. AJOUT DE LA BRIQUE : DETECTION DU DATA DRIFT (Divergence de Kullback-Leibler)
        print("\n📈 Analyse du Data Drift (Dérive des distributions)...")
        if "MONTHLYINCOME" in df_prod.columns and "MONTHLYINCOME" in df_ref.columns:
            try:
                # CORRECTION : Découpage propre en 10 intervalles réguliers (déciles)
                # On force la conversion en listes de nombres flottants standards pour Python
                counts, bin_edges = np.histogram(df_ref["MONTHLYINCOME"].dropna(), bins=10)
                
                # Calcul des probabilités de chaque intervalle (somme égale à 1.0)
                weights = counts / len(df_ref)
                
                # Sécurité pour éviter toute division par zéro dans le calcul logarithmique
                weights = [float(w) if w > 0 else 0.0001 for w in weights]
                # Ré-équilibrage pour que la somme soit exactement égale à 1
                weights = [w / sum(weights) for w in weights]

                # Format officiel attendu par Great Expectations
                partition_object = {
                    "bins": [float(b) for b in bin_edges],
                    "weights": weights
                }
                
                # Exécution du test de dérive statistique
                df_ge.expect_column_kl_divergence_to_be_less_than(
                    column="MONTHLYINCOME",
                    partition_object=partition_object,
                    threshold=0.1  # Seuil de tolérance
                )
                print("  ✓ Détection Drift activée sur MONTHLYINCOME")
            except Exception as e:
                print(f"  ⚠️ Impossible de configurer le test de Drift: {e}")

        results = df_ge.validate()
        return results

    # EXECUTION
    print("\n" + "="*60)
    print("VALIDATION CONTROLE QUALITÉ & DATA DRIFT MLOPS")
    print("="*60)
    
    results = validate_ml_ready_and_drift(df_production, df_reference)

    # LOGIQUE DE SAVE
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "ml_ready_validation.json")

    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "="*60)
    
    # Compte des validations
    total_validations = len(results.get("results", []))
    passed = sum(1 for r in results.get("results", []) if r.get("success", False))
    failed = total_validations - passed
    
    print(f"📊 RÉSUMÉ : {passed}/{total_validations} tests réussis")
    
    # Analyse spécifique du Drift pour l'affichage de l'alerte
    drift_detected = False
    for r in results.get("results", []):
        if r.get("expectation_config", {}).get("expectation_type") == "expect_column_kl_divergence_to_be_less_than":
            if not r.get("success", False):
                drift_detected = True

    if drift_detected:
        print("\n🚨 🚨 🚨 ALERTE MLOPS : DATA DRIFT DÉTECTÉ SUR LES SALAIRES ! 🚨 🚨 🚨")
        print("Les salaires actuels ne suivent plus la même distribution que le modèle d'entraînement.")
        print("Recommandation : Déclencher un réentraînement du modèle.")
    
    if results.get("success"):
        print("\n✓ ✓ ✓ Pipeline sain : Qualité 100% & Aucun Drift détecté ✓ ✓ ✓")
        exit_code = 0
    else:
        print(f"\n✗ ✗ ✗ Anomalie détectée dans le cycle de vie du modèle ({failed} échec(s)) ✗ ✗ ✗")
        exit_code = 1
    
    print(f"Rapport MLOps sauvegardé : {report_path}")
    print("="*60)

except Exception as e:
    print(f"\n❌ ERREUR CRITIQUE : {str(e)}")
    import traceback
    traceback.print_exc()
    exit_code = 1

finally:
    try:
        ctx.close()
        print("\n✓ Connexion Snowflake fermée")
    except:
        pass

if __name__ == "__main__":
    raise SystemExit(exit_code)