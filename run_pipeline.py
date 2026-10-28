"""
Complete End-to-End Pipeline Runner:
Executes the full pipeline for:
"XGBoost Classifier with Feature Interaction Constraints for Low-Signature Polymorphic Malware Detection"

Steps:
1. Data Acquisition (ClaMP PE malware benchmark dataset)
2. Preprocessing & Stratified Train/Test Split
3. Model Training (Random Forest, Standard XGBoost, Constrained XGBoost)
4. Evaluation & Visualization Suite (Confusion matrices, ROC curves, Metrics comparison)
5. Feature Interaction Constraints Audit & Domain Isolation Verification
"""
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.download_data import download_dataset
from src.preprocess import load_and_preprocess_data
from src.train import train_all_models
from src.evaluate import evaluate_models
from src.explain import analyze_tree_interactions


def run_pipeline():
    start_total = time.time()
    print("=" * 75)
    print("  XGBoost with Feature Interaction Constraints for Polymorphic Malware Detection")
    print("                       End-to-End Execution Pipeline")
    print("=" * 75)

    # Step 1: Download / Verify Dataset
    print("\n[STEP 1/5] Acquiring PE Malware Benchmark Dataset...")
    download_dataset()

    # Step 2: Data Preprocessing
    print("\n[STEP 2/5] Preprocessing, Feature Validation & Stratified Train/Test Splitting...")
    load_and_preprocess_data()

    # Step 3: Model Training
    print("\n[STEP 3/5] Training Classifiers (RF, Standard XGBoost, Constrained XGBoost)...")
    train_all_models()

    # Step 4: Model Evaluation & Visualizations
    print("\n[STEP 4/5] Evaluating Models & Generating Visualization Artifacts...")
    metrics = evaluate_models()

    # Step 5: Feature Interaction Constraint Verification
    print("\n[STEP 5/5] Auditing Tree Splits & Verifying Domain Isolation...")
    audit = analyze_tree_interactions()

    total_time = time.time() - start_total
    print("\n" + "=" * 75)
    print(f"  PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("=" * 75)
    print(f"\nKey Artifacts Created:")
    print(f"  - Clean Datasets:  data/processed/train.csv, data/processed/test.csv")
    print(f"  - Trained Models:  models/random_forest_baseline.joblib")
    print(f"                     models/xgboost_standard.joblib")
    print(f"                     models/xgboost_constrained.joblib")
    print(f"  - Metrics Summary: results/metrics_summary.json")
    print(f"  - Figures:         results/figures/confusion_matrices.png")
    print(f"                     results/figures/model_comparison_metrics.png")
    print(f"                     results/figures/roc_curves.png")
    print(f"                     results/figures/feature_importance.png")
    print(f"                     results/figures/feature_group_importance.png")
    print(f"                     results/figures/interaction_constraint_isolation.png")
    print(f"\nTo launch the interactive Streamlit interface, run:")
    print(f"  streamlit run app.py")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
