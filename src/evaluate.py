"""
Model Evaluation and Visualization Module:
- Evaluates Random Forest, Standard XGBoost, and Constrained XGBoost on the test set
- Computes comprehensive classification metrics:
    * Accuracy, Precision, Recall (Detection Rate), F1-Score, ROC-AUC, FPR
- Generates high-resolution visualization artifacts:
    * Confusion Matrix heatmaps
    * Multi-model metric comparison bar chart
    * ROC Curves with AUC scores
    * Precision-Recall curves
    * Feature importance comparisons color-coded by domain group
- Persists all empirical metrics to results/metrics_summary.json
"""
import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, average_precision_score
)

from src.config import (
    TEST_DATA_PATH, MODELS_DIR, RESULTS_DIR, FIGURES_DIR,
    METRICS_PATH, TARGET_COL, CLASS_NAMES, FEATURE_GROUPS
)


# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def evaluate_models():
    """Load models, evaluate on test set, save metrics and generate figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading test dataset from: {TEST_DATA_PATH}")
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].values

    # Load trained models
    model_files = {
        "Random Forest (Baseline)": MODELS_DIR / "random_forest_baseline.joblib",
        "Standard XGBoost": MODELS_DIR / "xgboost_standard.joblib",
        "Constrained XGBoost": MODELS_DIR / "xgboost_constrained.joblib"
    }

    models = {}
    for name, path in model_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}. Run src/train.py first!")
        models[name] = joblib.load(path)

    metrics_results = {}
    eval_data = {}

    print("\n" + "=" * 70)
    print(f"{'Model Evaluation Summary (Test Set, N = ' + str(len(y_test)) + ')':^70}")
    print("=" * 70)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        metrics_results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "false_positive_rate": round(float(fpr), 4),
            "false_negative_rate": round(float(fnr), 4),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            }
        }

        eval_data[name] = {
            "y_pred": y_pred,
            "y_prob": y_prob,
            "cm": cm
        }

        print(f"\n>>> {name}:")
        print(f"    - Accuracy:  {acc * 100:.2f}%")
        print(f"    - Precision: {prec * 100:.2f}%")
        print(f"    - Recall:    {rec * 100:.2f}% (Detection Rate)")
        print(f"    - F1-Score:  {f1:.4f}")
        print(f"    - ROC-AUC:   {roc_auc:.4f}")
        print(f"    - FPR:       {fpr * 100:.2f}% (False Alarm Rate)")
        print(f"    - Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Save metrics JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_results, f, indent=2)
    print(f"\n[+] Saved exact metrics to: {METRICS_PATH}")

    # =============================================================
    # 1. Visualization: Confusion Matrices Heatmap
    # =============================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    palette_cm = ["#2b5c8f", "#1e7e34", "#d9534f"]

    for idx, (name, data) in enumerate(eval_data.items()):
        cm = data["cm"]
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Annotations with counts and percentages
        labels = np.array([
            [f"TN: {cm[0,0]}\n({cm_norm[0,0]*100:.1f}%)", f"FP: {cm[0,1]}\n({cm_norm[0,1]*100:.1f}%)"],
            [f"FN: {cm[1,0]}\n({cm_norm[1,0]*100:.1f}%)", f"TP: {cm[1,1]}\n({cm_norm[1,1]*100:.1f}%)"]
        ])

        sns.heatmap(
            cm, annot=labels, fmt="", cmap="Blues", cbar=(idx == 2),
            ax=axes[idx], annot_kws={"size": 11, "weight": "bold"},
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
        )
        axes[idx].set_title(name, fontsize=12, pad=10, weight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        if idx == 0:
            axes[idx].set_ylabel("True Label", fontsize=10)

    fig.suptitle("Confusion Matrix Comparison on PE Malware Test Set (N=1042)", fontsize=14, weight="bold", y=1.03)
    plt.tight_layout()
    cm_path = FIGURES_DIR / "confusion_matrices.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved confusion matrix plot to: {cm_path}")

    # =============================================================
    # 2. Visualization: Metrics Comparison Bar Chart
    # =============================================================
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    chart_data = []
    for model_name, m in metrics_results.items():
        chart_data.append({
            "Model": model_name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"],
            "ROC-AUC": m["roc_auc"]
        })

    df_plot = pd.DataFrame(chart_data)
    df_melted = df_plot.melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    palette_models = {"Random Forest (Baseline)": "#5C6BC0", "Standard XGBoost": "#26A69A", "Constrained XGBoost": "#EF5350"}
    
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette=palette_models, ax=ax)
    ax.set_ylim(0.90, 1.01)  # Focus on top discrimination range
    ax.set_title("Performance Metrics Comparison Across Models", fontsize=13, weight="bold", pad=12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlabel("Evaluation Metric", fontsize=11)
    ax.legend(title="Model Architecture", loc="lower right", frameon=True)
    
    # Annotate bar values
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 3),
                        textcoords='offset points')

    plt.tight_layout()
    metrics_chart_path = FIGURES_DIR / "model_comparison_metrics.png"
    plt.savefig(metrics_chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved metrics comparison bar chart to: {metrics_chart_path}")

    # =============================================================
    # 3. Visualization: ROC Curves
    # =============================================================
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"Random Forest (Baseline)": "#5C6BC0", "Standard XGBoost": "#00897B", "Constrained XGBoost": "#E53935"}
    styles = {"Random Forest (Baseline)": "--", "Standard XGBoost": "-.", "Constrained XGBoost": "-"}

    for name, data in eval_data.items():
        fpr_curve, tpr_curve, _ = roc_curve(y_test, data["y_prob"])
        auc_val = metrics_results[name]["roc_auc"]
        ax.plot(
            fpr_curve, tpr_curve, label=f"{name} (AUC = {auc_val:.4f})",
            color=colors[name], linestyle=styles[name], linewidth=2.2
        )

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label="Random Guess (AUC = 0.5000)")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, weight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    roc_path = FIGURES_DIR / "roc_curves.png"
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved ROC curves to: {roc_path}")

    # =============================================================
    # 4. Visualization: Top Feature Importances (Standard vs Constrained)
    # =============================================================
    std_xgb = models["Standard XGBoost"]
    const_xgb = models["Constrained XGBoost"]
    feature_names = X_test.columns.tolist()

    # Map each feature to its domain group
    feature_to_group = {}
    for g_name, f_list in FEATURE_GROUPS.items():
        clean_name = g_name.replace("Group_", "").replace("_", " ")
        for f in f_list:
            feature_to_group[f] = clean_name

    std_imp = pd.Series(std_xgb.feature_importances_, index=feature_names)
    const_imp = pd.Series(const_xgb.feature_importances_, index=feature_names)

    top_std = std_imp.nlargest(12)
    top_const = const_imp.nlargest(12)

    group_colors = {
        "1 DOS COFF Header": "#3F51B5",
        "2 Optional Header Memory": "#009688",
        "3 Entropy Polymorphic": "#FF9800",
        "4 Packer Suspicious Sections": "#E91E63"
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Standard XGBoost Importance
    colors_std = [group_colors.get(feature_to_group.get(f, ""), "#757575") for f in top_std.index]
    axes[0].barh(top_std.index[::-1], top_std.values[::-1], color=colors_std[::-1], edgecolor="#333333", alpha=0.85)
    axes[0].set_title("Standard XGBoost\n(Unconstrained Feature Interactions)", fontsize=12, weight="bold")
    axes[0].set_xlabel("Relative Importance (Gain)", fontsize=10)

    # Constrained XGBoost Importance
    colors_const = [group_colors.get(feature_to_group.get(f, ""), "#757575") for f in top_const.index]
    axes[1].barh(top_const.index[::-1], top_const.values[::-1], color=colors_const[::-1], edgecolor="#333333", alpha=0.85)
    axes[1].set_title("Constrained XGBoost\n(Feature Interaction Constraints Enforced)", fontsize=12, weight="bold")
    axes[1].set_xlabel("Relative Importance (Gain)", fontsize=10)

    # Custom legend for domain groups
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='#333333', label=group)
                       for group, color in group_colors.items()]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05),
               title="Feature Functional Domain Group", fontsize=9, title_fontsize=10)

    fig.suptitle("Feature Importance Comparison: Standard vs. Constrained XGBoost", fontsize=14, weight="bold", y=1.10)
    plt.tight_layout()
    feat_path = FIGURES_DIR / "feature_importance.png"
    plt.savefig(feat_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved feature importance plot to: {feat_path}")

    # =============================================================
    # 5. Visualization: Feature Group Importance Distribution
    # =============================================================
    group_imp_records = []
    for model_name, model in [("Standard XGBoost", std_xgb), ("Constrained XGBoost", const_xgb)]:
        importances = model.feature_importances_
        for f, imp in zip(feature_names, importances):
            group_imp_records.append({
                "Model": model_name,
                "Domain Group": feature_to_group.get(f, "Other"),
                "Importance": imp
            })
    df_grp_imp = pd.DataFrame(group_imp_records).groupby(["Model", "Domain Group"])["Importance"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=df_grp_imp, x="Domain Group", y="Importance", hue="Model", palette=["#26A69A", "#EF5350"], ax=ax)
    ax.set_title("Total Feature Importance Allocated by Functional Domain", fontsize=12, weight="bold", pad=10)
    ax.set_ylabel("Cumulative Relative Importance", fontsize=10)
    ax.set_xlabel("Functional Domain Group", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    ax.legend(title="Model", frameon=True)
    plt.tight_layout()
    grp_path = FIGURES_DIR / "feature_group_importance.png"
    plt.savefig(grp_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved group importance chart to: {grp_path}")

    print("\n[+] Model evaluation and visualization suite completed successfully!")
    return metrics_results


if __name__ == "__main__":
    evaluate_models()
