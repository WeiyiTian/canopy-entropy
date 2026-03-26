import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    data_dir = os.getenv("DATA_DIR")
    model_dir = os.getenv("MODEL_DIR")
    outputs_dir = os.getenv("OUTPUTS_DIR")
    results_dir = os.getenv("RESULTS_DIR")

settings = Settings()
