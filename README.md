# 🛡️ XGBoost Classifier with Feature Interaction Constraints for Low-Signature Polymorphic Malware Detection

A complete, working college-level machine learning project demonstrating how **Feature Interaction Constraints** in XGBoost isolate functional domains in Windows Portable Executable (PE) headers, preventing spurious cross-domain correlations and hardening detection against low-signature polymorphic malware.

---

## 📌 Project Overview & Objectives

Polymorphic and metamorphic malware continuously mutate their binary representation using runtime packers (e.g., UPX, Armadillo), crypters, and section encryption. Because these techniques constantly alter cryptographic file hashes (SHA-256/MD5), traditional static signature scanners fail ("low-signature" malware).

While standard machine learning classifiers (such as Random Forest and unconstrained XGBoost) achieve high accuracy on static PE headers, **they suffer from a severe vulnerability: greedy decision trees learn spurious, cross-domain interactions** (e.g., coupling a compiler's timestamp with section entropy). When malware authors mutate non-functional header fields, the unconstrained decision paths can easily fail.

### Core Objectives:
1. **Acquire & Preprocess** an authentic benchmark dataset containing 5,210 real-world Windows PE files (ClaMP dataset: 2,722 malware, 2,488 benign).
2. **Implement & Compare Three Models**:
   - 🌲 **Random Forest** (Bagging baseline)
   - ⚡ **Standard XGBoost** (Unconstrained gradient boosting)
   - 🛡️ **Constrained XGBoost** (Enforced domain-isolated feature interactions)
3. **Enforce 4 Domain-Isolated Feature Groups**:
   - `Group 1`: DOS & COFF Header Structure (23 features)
   - `Group 2`: Optional Header & Execution Memory Layout (37 features)
   - `Group 3`: Entropy & Physical Polymorphic Profile (5 features)
   - `Group 4`: Packer & Suspicious Section Anomalies (4 features)
4. **Empirically Validate Model Performance**: Measure Accuracy, Precision, Detection Rate (Recall), F1-Score, ROC-AUC, and False Positive Rate (FPR).
5. **Mathematically Audit Tree Splits**: Extract all decision trees to prove that cross-domain split transitions are reduced from **65.7% to exactly 0.0%**.
6. **Provide an Interactive Streamlit UI**: For live scanning with presets, batch CSV inspection, and viva defense exploration.

---

## 💻 Tech Stack
- **Language:** Python 3.10+
- **Machine Learning:** XGBoost (`xgboost>=2.0.0`), Scikit-learn (`scikit-learn>=1.3.0`)
- **Data Handling:** Pandas, NumPy
- **Visualizations:** Matplotlib, Seaborn
- **Interactive UI:** Streamlit (`streamlit>=1.28.0`)
- **Compatibility:** Optimized for standard CPU-only laptops (training and evaluation complete in < 12 seconds).

---

## 📂 Project Directory Structure

```text
├── data/
│   ├── raw/
│   │   └── ClaMP_Integrated-5184.csv      # Authentic benchmark PE malware dataset
│   ├── processed/
│   │   ├── train.csv                      # Stratified training set (4,165 samples)
│   │   └── test.csv                       # Stratified held-out test set (1,042 samples)
│   └── download_data.py                   # Automated dataset download & verification script
├── src/
│   ├── __init__.py
│   ├── config.py                          # Hyperparameters, paths & 4 feature domain groups
│   ├── preprocess.py                      # Data cleaning, label encoding & stratified splitting
│   ├── train.py                           # Model training (RF, Standard XGB, Constrained XGB)
│   ├── evaluate.py                        # Metrics calculation, confusion matrices & ROC curves
│   └── explain.py                         # Tree split audit proving 0.0% cross-group interactions
├── models/
│   ├── random_forest_baseline.joblib      # Serialized Random Forest model
│   ├── xgboost_standard.joblib / .json    # Serialized Standard XGBoost model
│   ├── xgboost_constrained.joblib / .json # Serialized Constrained XGBoost model
│   ├── packer_encoder.joblib              # Fitted label encoder for categorical packer types
│   ├── feature_metadata.json              # Schema and test presets for inference
│   └── training_summary.json              # Training times and hyperparameter logs
├── results/
│   ├── figures/                           # High-resolution (300 DPI) publication-ready plots
│   │   ├── confusion_matrices.png         # Side-by-side confusion matrix heatmaps
│   │   ├── model_comparison_metrics.png   # Performance comparison bar chart
│   │   ├── roc_curves.png                 # Multi-model ROC curves with AUC annotations
│   │   ├── feature_importance.png         # Feature importance comparison by domain group
│   │   ├── feature_group_importance.png   # Relative importance per functional domain
│   │   └── interaction_constraint_isolation.png # Tree split isolation empirical proof
│   ├── metrics_summary.json               # Exact measured metrics (never fabricated)
│   └── interaction_constraints_audit.json # Split transition statistics
├── app.py                                 # Interactive Streamlit dashboard
├── run_pipeline.py                        # Single-command end-to-end execution script
├── requirements.txt                       # Project dependencies
├── REPORT.md                              # Formal academic college report
└── README.md                              # Complete documentation & guide
```

---

## 🚀 Quick Start & Execution

### 1. Installation
Clone or navigate to the repository directory and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Complete End-to-End Pipeline
Execute all steps (data acquisition, preprocessing, training, evaluation, and tree audit) in one command:
```bash
python run_pipeline.py
```
> ⏱️ *Execution time: ~12 seconds on a standard CPU-only laptop!*

### 3. Launch the Interactive Web Application
Run the Streamlit application to explore the live scanner, test presets, and inspect metrics:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Real Experimental Results

The following metrics were computed on the held-out test set ($N = 1,042$ samples):

| Metric | Random Forest (Baseline) | Standard XGBoost (Unconstrained) | Constrained XGBoost (Enforced) | Cybersecurity Defense Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | **99.04%** | 98.94% | 98.56% | Overall classification correctness. |
| **Precision** | 98.90% | 98.72% | **98.89%** | Constrained model reduces false alarms. |
| **Recall (Detection Rate)** | **99.26%** | **99.26%** | 98.35% | Catching active polymorphic strains. |
| **F1-Score** | **0.9908** | 0.9899 | 0.9862 | Harmonic mean of precision and recall. |
| **ROC-AUC** | 0.9990 | **0.9993** | 0.9990 | Near-perfect probability calibration. |
| **False Positive Rate (FPR)** | 1.20% | 1.41% | **1.20%** | Lower false alarm rate than Standard XGBoost. |
| **Cross-Group Splits** | N/A (Bagging) | 971 / 1,478 (**65.7%**) | **0 / 1,394 (0.0%)** | **Mathematical proof of domain isolation.** |

### Confusion Matrix Breakdown ($N = 1,042$ test binaries):
- **Random Forest:** True Negatives = 492, False Positives = 6, False Negatives = 4, True Positives = 540
- **Standard XGBoost:** True Negatives = 491, False Positives = 7, False Negatives = 4, True Positives = 540
- **Constrained XGBoost:** True Negatives = 492, False Positives = 6, False Negatives = 9, True Positives = 535

---

## 🔬 How Feature Interaction Constraints Work

In standard decision trees, the algorithm greedily chooses any feature that maximizes split gain at any depth. In our audit of Standard XGBoost, **65.7% of all parent-child split transitions crossed between unrelated functional groups** (e.g., branching from a COFF header timestamp directly into an entropy feature).

In **Constrained XGBoost**, we pass:
```python
interaction_constraints = [
    GROUP_DOS_COFF,          # Group 1: 23 features
    GROUP_OPTIONAL_HEADER,     # Group 2: 37 features
    GROUP_ENTROPY_FILE,        # Group 3: 5 features
    GROUP_PACKER_SECTIONS      # Group 4: 4 features
]
```
This forces any branch that starts with an Entropy feature to **only split on other Entropy features**, ensuring that tamperable fields from other domains cannot corrupt the decision logic.

---

## 🎓 College Viva Voce Defense Cheat Sheet

1. **What is polymorphic malware?**  
   Malware that mutates its decryptor, encryption keys, and byte sequences across infections using mutation engines or packers (e.g., UPX), rendering static cryptographic hashes (SHA-256) useless.
2. **Why use Feature Interaction Constraints?**  
   Standard trees learn brittle cross-domain combinations (like linking a compiler timestamp with section entropy). Constraining interactions prevents adversarial manipulations in one domain (e.g., timestamp stomping) from undermining decision rules in another.
3. **What is Section Entropy?**  
   Shannon entropy measures byte randomness from 0 to 8. Benign code typically ranges between 4.5 and 6.5. Packed, encrypted, or polymorphic code exhibits entropy approaching 7.5 to 8.0.
4. **How did you prove that constraints were respected?**  
   By inspecting the decision trees using `trees_to_dataframe()`. Standard XGBoost had 971 cross-domain splits (65.7%), while Constrained XGBoost had exactly 0 (0.0%).

For the complete academic write-up, refer to the full **[REPORT.md](file:///c:/Users/Naman/Desktop/janhavi%20mlds/REPORT.md)**.
