import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.image import random_flip_up_down, random_flip_left_right, random_contrast, random_brightness, resize, rot90
from typing import Literal
import os
import numpy as np

def load_original_images(paths: dict, color_mode: Literal['grayscale','rgb']):
    req_path_keys = {'train_tumor','train_notumor','validation_tumor','validation_notumor'}
    if not req_path_keys.issubset(paths.keys()):
        raise ValueError(f"1 or more of the following path names are missing from the provided path dictionary: {', '.join(req_path_keys)}")

    num_train_tumor = len([file for file in os.listdir(paths['train_tumor']) if '.jpg' in file])
    num_train_notumor = len([file for file in os.listdir(paths['train_notumor']) if '.jpg' in file])
    num_validation_tumor = len([file for file in os.listdir(paths['validation_tumor']) if '.jpg' in file])
    num_validation_notumor = len([file for file in os.listdir(paths['validation_notumor']) if '.jpg' in file])
    print(num_train_tumor, num_train_notumor, num_validation_tumor, num_validation_notumor)
    
    train_tumor = image_dataset_from_directory(directory = paths['train_tumor'], color_mode = color_mode, labels = [1.0]*num_train_tumor)
    train_notumor = image_dataset_from_directory(directory = paths['train_notumor'], color_mode = color_mode, labels = [0.0]*num_train_notumor)
    validation_tumor = image_dataset_from_directory(directory = paths['validation_tumor'], color_mode = color_mode, labels = [1.0]*num_validation_tumor)
    validation_notumor = image_dataset_from_directory(directory = paths['validation_notumor'], color_mode = color_mode, labels = [0.0]*num_validation_notumor)
    return train_tumor, train_notumor, validation_tumor, validation_notumor

def create_model_datasets(train_tumor, train_notumor, validation_tumor, validation_notumor, img_height, img_width):
    AUTOTUNE = tf.data.AUTOTUNE
    def process_images(image, label, transformation: Literal['preprocess_only','preprocess_and_augment']):
        def random_val() -> float:
            return np.random.random()
        def random_turns() -> int:
            return np.random.randint(low = 1, high = 4)
        image = resize(image, size = (img_height, img_width), method = 'bilinear')
        if transformation == 'preprocess_and_augment':
            if random_val() > 0.5:
                image = random_flip_left_right(image)
            if random_val() > 0.5:
                image = random_flip_up_down(image)
            if random_val() > 0.5:
                image = rot90(image, k = random_turns())
            if random_val() > 0.8:
                image = random_contrast(image, 0.2, 0.5)
            if random_val() > 0.8:
                image = random_brightness(image, 0.2)
        return image, label
    
    # Processing of training data
    tumor_original = train_tumor.map(
        lambda img, lab: process_images(img, lab, 'preprocess_only')
    )
    tumor_augmented = train_tumor.map(
        lambda img, lab: process_images(img, lab, 'preprocess_and_augment')
    ).repeat(1)

    notumor_original = train_notumor.map(
        lambda img, lab: process_images(img, lab, 'preprocess_only')
    )
    notumor_augmented = train_notumor.map(
        lambda img, lab: process_images(img, lab, 'preprocess_and_augment')
    ).repeat(5)

    final_tumor_train = tumor_original.concatenate(tumor_augmented)
    final_notumor_train = notumor_original.concatenate(notumor_augmented)
    final_train = final_tumor_train.concatenate(final_notumor_train)

    # Processing of validation data (NO augmentation)
    validation_tumor = validation_tumor.map(
        lambda img, lab: process_images(img, lab, 'preprocess_only')
    )
    validation_notumor = validation_notumor.map(
        lambda img, lab: process_images(img, lab, 'preprocess_only')
    )
    final_validation = validation_tumor.concatenate(validation_notumor)

    # Shuffle and prefetch data 
    final_train = final_train.shuffle(buffer_size = 1000).prefetch(buffer_size = AUTOTUNE)
    final_validation = final_validation.prefetch(buffer_size = AUTOTUNE)

    return final_train, final_validation

def resize_test_image(path, img_height, img_width):
    raw_img = tf.keras.preprocessing.image.load_img(path)
    resized_img = resize(raw_img, size = (img_height, img_width), method = 'bilinear')
    return resized_img