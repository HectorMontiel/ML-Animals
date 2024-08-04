import tensorflow as tf
from data_preprocessing import create_data_generators
from model import create_model

def train_model(train_dir, validation_dir, epochs=20, batch_size=32):
    # Crear generadores de datos
    train_generator, validation_generator = create_data_generators(train_dir, validation_dir, batch_size=batch_size)

    # Configurar el modelo
    input_shape = (150, 150, 3)
    num_classes = len(train_generator.class_indices)
    model = create_model(input_shape, num_classes)

    # Configurar el punto de control para guardar el modelo
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        'D:/Documentos/Programs/ML-Animals/ML-Animals/data/best_model.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    # Configurar la detención temprana para evitar sobreajuste
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    )

    # Ajustar el modelo
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )

    return model, history

if __name__ == "__main__":
    train_dir = 'D:/Documentos/Programs/ML-Animals/ML-Animals/data/images/train'
    validation_dir = 'D:/Documentos/Programs/ML-Animals/ML-Animals/data/images/validation'
    model, history = train_model(train_dir, validation_dir, epochs=20)
