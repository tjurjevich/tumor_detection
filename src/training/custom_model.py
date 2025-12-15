from src.utils.image_processing import load_original_images, create_model_datasets
from src.utils.models import CustomTumorClassifier, TLTumorClassifier
import logging
import yaml
import os
from pathlib import Path
import datetime

with open('config/params.yaml', 'r') as f:
    config = yaml.safe_load(f)


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(f"logs/log_test_{datetime.date.today().strftime('%Y-%m-%d')}.log")
    ]
)