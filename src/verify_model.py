import tensorflow as tf

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        # Imprimir la estructura del modelo para verificar
        model.summary()
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    model_path = 'D:/Documentos/Programs/ML-Animals/ML-Animals/data/best_model.keras'
    load_model(model_path)
