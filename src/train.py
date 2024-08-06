import os
import keras_tuner as kt
from data_preprocessing import create_data_generators
from model import build_model

def get_num_classes(train_dir):
    train_generator, _ = create_data_generators(train_dir, None, batch_size=1)
    return len(train_generator.class_indices)

def train_model(train_dir, validation_dir):
    if not isinstance(train_dir, str) or not os.path.isdir(train_dir):
        raise ValueError("train_dir must be a valid directory path")
    if validation_dir is not None and (not isinstance(validation_dir, str) or not os.path.isdir(validation_dir)):
        raise ValueError("validation_dir must be a valid directory path or None")
        
    num_classes = get_num_classes(train_dir)
    train_generator, validation_generator = create_data_generators(train_dir, validation_dir, batch_size=32)

    def model_builder(hp):
        return build_model(hp, num_classes)

    tuner = kt.Hyperband(
        model_builder,
        objective='val_accuracy',
        max_epochs=20,
        hyperband_iterations=2,
        directory='kt_dir',
        project_name='image_classification'
    )

    if validation_generator:
        tuner.search(
            train_generator,
            epochs=20,
            validation_data=validation_generator
        )
    else:
        tuner.search(
            train_generator,
            epochs=20
        )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save('C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/best_model.h5')

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: {best_hps.values}")

    final_accuracy = best_hps.values.get('val_accuracy', 'N/A')
    print(f'Final Validation Accuracy: {final_accuracy:.2f}%')

    return best_model

if __name__ == "__main__":
    train_dir = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/images/train'
    validation_dir = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/images/validation'
    
    if not os.path.isdir(train_dir):
        raise ValueError(f"The provided train directory path is not valid: {train_dir}")
    if validation_dir is not None and not os.path.isdir(validation_dir):
        raise ValueError(f"The provided validation directory path is not valid: {validation_dir}")
        
    model = train_model(train_dir, validation_dir)
