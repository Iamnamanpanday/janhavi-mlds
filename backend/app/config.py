from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT / "artifacts"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024
SCAN_TIMEOUT = int(os.getenv("SCAN_TIMEOUT_SECONDS", "8"))
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
DB_PATH = os.getenv("DATABASE_URL", "sqlite:///./scan_history.db").replace("sqlite:///", "")
