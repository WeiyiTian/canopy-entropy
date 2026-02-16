import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    model_path = os.getenv("MODEL_PATH")
    results_path = os.getenv("RESULTS_PATH")

settings = Settings()