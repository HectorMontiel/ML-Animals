import tensorflow as tf
from data_preprocessing import create_data_generators
from model import create_model

def train_model(train_dir, validation_dir, epochs=20, batch_size=32):
    train_generator, validation_generator = create_data_generators(train_dir, validation_dir, batch_size)

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

    h5_model_path = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/best_model.h5'
    model.save(h5_model_path)
    print(f'Modelo guardado en {h5_model_path}')

    return model, history

if __name__ == "__main__":
    train_dir = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/images/train'
    validation_dir = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/images/validation'
    model, history = train_model(train_dir, validation_dir, epochs=20)
