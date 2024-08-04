import tensorflow as tf
from data_preprocessing import create_data_generators
from model import create_model

def train_model(train_dir, validation_dir, epochs=20, batch_size=32):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical'
    )

    input_shape = (150, 150, 3)
    model = create_model(input_shape, num_classes=len(train_generator.class_indices))

    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps
    )

    model.save('C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/best_model.keras')
    return model, history

if __name__ == "__main__":
    train_dir = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/images/train'
    validation_dir = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/images/validation'
    model, history = train_model(train_dir, validation_dir, epochs=20)
