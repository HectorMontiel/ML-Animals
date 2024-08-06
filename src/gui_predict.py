import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
import json

def load_and_preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def load_class_indices(class_indices_path):
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    # Invertir el mapeo para obtener nombres de clases a partir de los índices
    class_indices = {v: k for k, v in class_indices.items()}
    return class_indices

def predict_image(model_path, class_indices_path, img_path):
    model = tf.keras.models.load_model(model_path)
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    class_indices = load_class_indices(class_indices_path)
    predicted_class = class_indices.get(predicted_class_index, "Unknown")
    return predicted_class

def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_class = predict_image(model_path, class_indices_path, file_path)
        result_label.config(text=f'Predicted Class: {predicted_class}')

if __name__ == "__main__":
    model_path = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/best_model.keras'
    class_indices_path = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/class_indices.json'

    root = tk.Tk()
    root.title("Image Classifier")

    Label(root, text="Select an image to classify:").pack(pady=10)
    Button(root, text="Browse", command=open_file_dialog).pack(pady=10)
    result_label = Label(root, text="")
    result_label.pack(pady=10)

    root.mainloop()
