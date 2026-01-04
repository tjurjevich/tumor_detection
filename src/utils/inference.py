import tensorflow as tf
from src.utils.models import CustomTumorClassifier
from tensorflow.image import resize
from typing import Literal

# Will return an 3-channel array representation of the selected image
def process_test_image(method: Literal["file_path","array"], img_height: int, img_width: int, img_path = None, img_array = None):
    if method == "file_path":
        if img_path is None:
            raise ValueError("img_path is required based on selected method of 'file_path'")
        raw_img = tf.keras.preprocessing.image.load_img(img_path)
        image = resize(raw_img, size = (img_height, img_width), method = 'bilinear')
        return tf.expand_dims(image, axis=0)
    else:
        image = resize(img_array, size = (img_height, img_width), method = 'bilinear')
        return tf.expand_dims(image, axis=0)

# Loads model into the environment
def load_model(model_path):
    mod = tf.keras.models.load_model(
        model_path
    )
    return mod

# Makes prediction for the selected image
def make_prediction(model, image):
    prediction = model.predict(image, verbose = 0)
    if prediction < 1e-4:
        return 'No tumor detected (<0.01%)'
    if prediction >= 1e-4 and prediction < 0.5:
        return f'No tumor detected ({round((prediction[0][0])*100, 3)}%)'
    if prediction >= 0.5 and prediction < 0.9999:
        return f'Tumor detected ({round((prediction[0][0])*100, 3)}%)'
    if prediction >= 0.9999:
        return f'Tumor detected (>99.99%)'
    
# img = process_test_image('data/testing/tumor/Tr-me_0010.jpg', 256, 256)
# mod = load_model('saved_models/custom_model.keras')
# pred = make_prediction(mod, img)