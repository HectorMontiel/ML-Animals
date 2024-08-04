import os
import shutil
from sklearn.model_selection import train_test_split

# Define el path a la carpeta de imágenes
images_dir = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/images.tar/images'  # Actualiza la ruta según la ubicación correcta
output_dir = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/images'

# Lista todas las imágenes y asume que los nombres de las imágenes contienen el nombre de la raza
all_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

# Supongamos que el nombre de la raza está en el nombre del archivo
breeds = set()
for img in all_images:
    breed = img.split('_')[0]  # Cambia esto según cómo están organizadas las imágenes
    breeds.add(breed)

# Crea directorios para entrenamiento y validación
train_dir = os.path.join(output_dir, 'train')
validation_dir = os.path.join(output_dir, 'validation')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Crea subdirectorios para cada raza
for breed in breeds:
    os.makedirs(os.path.join(train_dir, breed), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, breed), exist_ok=True)

# Divide las imágenes en conjuntos de entrenamiento y validación
train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)

def move_images(images, target_dir):
    for img in images:
        breed = img.split('_')[0]
        src_path = os.path.join(images_dir, img)
        dst_path = os.path.join(target_dir, breed, img)
        shutil.copy(src_path, dst_path)

move_images(train_images, train_dir)
move_images(val_images, validation_dir)

print("Organización completa.")
