import tensorflow as tf
import json
from data_preprocessing import create_data_generators

def load_class_indices(mapping_path):
    with open(mapping_path, 'r') as f:
        class_indices = json.load(f)
    return class_indices

def evaluate_model(model_path, validation_dir, mapping_path):
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    _, validation_generator = create_data_generators(validation_dir, validation_dir, batch_size=32)
    
    loss, accuracy = model.evaluate(validation_generator)
    print(f'Validation accuracy: {accuracy:.4f}')
    print(f'Validation loss: {loss:.4f}')

if __name__ == "__main__":
    model_path = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/best_model.keras'
    validation_dir = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/images/validation'
    mapping_path = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/class_indices.json'
    
    evaluate_model(model_path, validation_dir, mapping_path)
