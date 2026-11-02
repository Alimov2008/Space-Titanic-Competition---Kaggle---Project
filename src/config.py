from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

DATA_DIR = PROJECT_ROOT / "data" / "spaceship-titanic"

TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

TARGET_COLUMN = "Transported"
RANDOM_STATE = 42
TEST_SIZE = 0.20


MODEL_PATH = MODEL_DIR / "best_pipeline.joblib"
METRICS_PATH = REPORT_DIR / "test_metrics.json"
