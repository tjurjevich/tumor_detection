from src.utils.image_processing import load_original_images, create_model_datasets
from src.utils.models import CustomTumorClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
import tensorflow as tf
import logging
import yaml
import os
import datetime

# load parameters
with open('config/params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# setup logging to terminal and also write to file
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(f"logs/custom_model_general_logs_{datetime.date.today().strftime('%Y-%m-%d')}.log")
    ]
)

# Set paths for data
data_paths = {
    'train_tumor': params["data"]["train_dir_tumor"],
    'train_notumor': params["data"]["train_dir_notumor"],
    'validation_tumor': params["data"]["validation_dir_tumor"],
    'validation_notumor': params["data"]["validation_dir_notumor"]
}

# Load data
logging.info("Loading original images...")
train_tumor, train_notumor, validation_tumor, validation_notumor = load_original_images(
    paths = data_paths,
    color_mode = params["data"]["color_mode"]
)
logging.info("Original images loaded successfully.")

# Process data for model training
logging.info("Transforming and augmenting data for model training...")
train_data, validation_data = create_model_datasets(
    train_tumor = train_tumor,
    train_notumor = train_notumor,
    validation_tumor = validation_tumor,
    validation_notumor = validation_notumor,
    img_height = params["data"]["image_height"],
    img_width = params["data"]["image_width"]
)
logging.info("Training and validation datasets successfully created.")

# Define callbacks to use during traning
early_stopping = EarlyStopping(
        monitor = "val_loss",
        min_delta = params["training"]["callbacks"]["early_stopping"]["min_delta"],
        patience = params["training"]["callbacks"]["early_stopping"]["patience"],
        start_from_epoch = params["training"]["callbacks"]["early_stopping"]["epoch_start"],
        verbose = 1,    
        mode = "min",
        restore_best_weights = True
)

lr_reducer = ReduceLROnPlateau(
    monitor = "val_loss",
    factor = params["training"]["callbacks"]["lr_plateau"]["factor"],
    patience = params["training"]["callbacks"]["lr_plateau"]["patience"],
    min_lr = params["training"]["callbacks"]["lr_plateau"]["min_lr"]
)

csv_logger = CSVLogger(
    filename = f"logs/custom_model_training_logs_{datetime.date.today().strftime('%Y-%m-%d')}.csv"
)
logging.info("Callbacks defined")

callbacks = [
    early_stopping,
    lr_reducer,
    csv_logger
]

# Initiate custom model
logging.info("Creating model definition")
model = CustomTumorClassifier(
    conv_layer_filters = params["model"]["custom"]["conv_filters"],
    pool_type = params["model"]["custom"]["pool_type"],
    dense_layer_units = params["model"]["custom"]["dense_units"],
    dropout_pct = params["model"]["custom"]["dropout_pct"]
)

logging.info("Compiling model")
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy','recall']  
)

logging.info("Beginning model training...")
model_history = model.fit(
    train_data, 
    validation_data = validation_data, 
    epochs = params["training"]["epochs"], 
    batch_size = params["training"]["batch_size"],
    callbacks = callbacks
)
logging.info("Model training complete")


model.save(params["model"]["custom"]["saved_model_dir"]) # save the model if desired
logging.info(f"Model architecturs and weights saved to {params["model"]["custom"]["saved_model_dir"]}")