import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model_path, validation_dir):
    # Asegurarnos de que el modelo se carga con la forma de entrada correcta
    model = tf.keras.models.load_model(model_path, compile=False)
    model.build(input_shape=(None, 150, 150, 3))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    validation_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    

    # Depuración: Imprimir la forma de los datos
    for data_batch, labels_batch in validation_generator:
        print(f'Data batch shape: {data_batch.shape}')
        print(f'Labels batch shape: {labels_batch.shape}')
        break  # Solo queremos imprimir la forma de los datos una vez

    loss, accuracy = model.evaluate(validation_generator)
    print(f'Validation Accuracy: {accuracy}')
    print(f'Validation Loss: {loss}')

if __name__ == "__main__":
    model_path = 'D:/Documentos/Programs/ML-Animals/ML-Animals/data/best_model.keras'
    validation_dir = 'D:/Documentos/Programs/ML-Animals/ML-Animals/data/images/validation'
    evaluate_model(model_path, validation_dir)
