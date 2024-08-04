import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model_path, validation_dir):
    model = tf.keras.models.load_model(model_path)
    
    validation_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    loss, accuracy = model.evaluate(validation_generator)
    print(f'Validation Accuracy: {accuracy}')
    print(f'Validation Loss: {loss}')

if __name__ == "__main__":
    model_path = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/best_model.keras'
    validation_dir = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/images/validation'
    evaluate_model(model_path, validation_dir)
