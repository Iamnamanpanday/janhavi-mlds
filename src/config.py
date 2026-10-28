"""
Configuration module for the Malware Detection Project:
"XGBoost Classifier with Feature Interaction Constraints for Low-Signature Polymorphic Malware Detection"
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "ClaMP_Integrated-5184.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_PATH = RESULTS_DIR / "metrics_summary.json"

# Random Seed for Reproducibility
RANDOM_SEED = 42
TEST_SIZE = 0.20

# Feature Domain Groups
# These groups separate distinct functional domains of Windows PE executables.
# Restricting cross-group interactions prevents the model from learning brittle,
# spurious combinations that polymorphic/metamorphic malware manipulate.

GROUP_DOS_COFF = [
    'e_cblp', 'e_cp', 'e_cparhdr', 'e_maxalloc', 'e_sp', 'e_lfanew',
    'NumberOfSections', 'CreationYear',
    'FH_char0', 'FH_char1', 'FH_char2', 'FH_char3', 'FH_char4', 'FH_char5',
    'FH_char6', 'FH_char7', 'FH_char8', 'FH_char9', 'FH_char10', 'FH_char11',
    'FH_char12', 'FH_char13', 'FH_char14'
]

GROUP_OPTIONAL_HEADER = [
    'MajorLinkerVersion', 'MinorLinkerVersion', 'SizeOfCode', 'SizeOfInitializedData',
    'SizeOfUninitializedData', 'AddressOfEntryPoint', 'BaseOfCode', 'BaseOfData',
    'ImageBase', 'SectionAlignment', 'FileAlignment', 'MajorOperatingSystemVersion',
    'MinorOperatingSystemVersion', 'MajorImageVersion', 'MinorImageVersion',
    'MajorSubsystemVersion', 'MinorSubsystemVersion', 'SizeOfImage', 'SizeOfHeaders',
    'CheckSum', 'Subsystem',
    'OH_DLLchar0', 'OH_DLLchar1', 'OH_DLLchar2', 'OH_DLLchar3', 'OH_DLLchar4',
    'OH_DLLchar5', 'OH_DLLchar6', 'OH_DLLchar7', 'OH_DLLchar8', 'OH_DLLchar9', 'OH_DLLchar10',
    'SizeOfStackReserve', 'SizeOfStackCommit', 'SizeOfHeapReserve', 'SizeOfHeapCommit',
    'LoaderFlags'
]

GROUP_ENTROPY_FILE = [
    'E_text', 'E_data', 'filesize', 'E_file', 'fileinfo'
]

GROUP_PACKER_SECTIONS = [
    'sus_sections', 'non_sus_sections', 'packer', 'packer_type'
]

FEATURE_GROUPS = {
    "Group_1_DOS_COFF_Header": GROUP_DOS_COFF,
    "Group_2_Optional_Header_Memory": GROUP_OPTIONAL_HEADER,
    "Group_3_Entropy_Polymorphic": GROUP_ENTROPY_FILE,
    "Group_4_Packer_Suspicious_Sections": GROUP_PACKER_SECTIONS,
}

# XGBoost interaction constraints: list of lists of feature names
INTERACTION_CONSTRAINTS = [
    GROUP_DOS_COFF,
    GROUP_OPTIONAL_HEADER,
    GROUP_ENTROPY_FILE,
    GROUP_PACKER_SECTIONS
]

TARGET_COL = "class"
CLASS_NAMES = ["Benign", "Malware"]

# Model Hyperparameters
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 14,
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

XGB_STANDARD_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

XGB_CONSTRAINED_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "interaction_constraints": INTERACTION_CONSTRAINTS,
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}
