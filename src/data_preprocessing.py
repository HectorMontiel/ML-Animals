import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(train_dir, validation_dir, batch_size):
    if not isinstance(train_dir, str) or not os.path.isdir(train_dir):
        raise ValueError("train_dir must be a valid directory path")
    if validation_dir is not None and (not isinstance(validation_dir, str) or not os.path.isdir(validation_dir)):
        raise ValueError("validation_dir must be a valid directory path or None")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = None
    if validation_dir:
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='categorical'
        )

    return train_generator, validation_generator
