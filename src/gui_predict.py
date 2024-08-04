import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button

def load_and_preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image(model_path, img_path):
    model = tf.keras.models.load_model(model_path)
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class

def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_class = predict_image(model_path, file_path)
        result_label.config(text=f'Predicted Class: {predicted_class}')

if __name__ == "__main__":
    model_path = 'D:/Documentos/Programs/ML-Animals/ML-Animals/data/best_model.keras'

    root = tk.Tk()
    root.title("Image Classifier")

    Label(root, text="Select an image to classify:").pack(pady=10)
    Button(root, text="Browse", command=open_file_dialog).pack(pady=10)
    result_label = Label(root, text="")
    result_label.pack(pady=10)

    root.mainloop()
