import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    BRIGHT_DATA_API_TOKEN = os.getenv("BRIGHT_DATA_API_TOKEN", "bright_data_api_token")
    BRIGHT_DATA_DATASET_ID = "gd_lwxmeb2u1cniijd7t4"
    OUTPUT_DIR = "output/"
    MAX_CONCURRENT_VALIDATIONS = 3
    VALIDATION_DELAY_MIN = 1.5
    VALIDATION_DELAY_MAX = 3.5
    MAX_SNAPSHOT_WAIT = 600
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)

  # X.com Authentication Configuration
    X_LOGIN = os.getenv("X_LOGIN", "margati@ac.sce.ac.il")
    X_PASSWORD = os.getenv("X_PASSWORD", "15092025")
    X_BACKUP_LOGIN = os.getenv("X_BACKUP_LOGIN", "testbuba23@gmail.com")
    X_BACKUP_PASSWORD = os.getenv("X_BACKUP_PASSWORD", "15092025bubatest1")