import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    BRIGHT_DATA_API_TOKEN = os.getenv("BRIGHT_DATA_API_TOKEN", "357e781135f8ac1e81a9f3b2c23b0ca71778eeb6f29b4dc48c843415d679e70d")
    BRIGHT_DATA_DATASET_ID = "gd_lwxmeb2u1cniijd7t4"
    OUTPUT_DIR = "output/"
    MAX_CONCURRENT_VALIDATIONS = 3
    VALIDATION_DELAY_MIN = 1.5
    VALIDATION_DELAY_MAX = 3.5
    MAX_SNAPSHOT_WAIT = 600
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)
