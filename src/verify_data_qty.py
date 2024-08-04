import os

def count_images_in_directory(directory):
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                count += 1
    return count

train_dir = 'D:/Documentos/Programs/ML-Animals/ML-Animals/data/images/train'
validation_dir = 'D:/Documentos/Programs/ML-Animals/ML-Animals/data/images/validation'

print(f'Number of training images: {count_images_in_directory(train_dir)}')
print(f'Number of validation images: {count_images_in_directory(validation_dir)}')
