import tensorflow as tf

def evaluate_model(model_path, validation_dir, batch_size=32):
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical'
    )

    model = tf.keras.models.load_model(model_path)

    loss, accuracy = model.evaluate(validation_generator)
    print(f'Validation Loss: {loss}')
    print(f'Validation Accuracy: {accuracy}')

if __name__ == "__main__":
    model_path = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/best_model.h5'
    validation_dir = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/images/validation'
    evaluate_model(model_path, validation_dir)
