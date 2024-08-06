import os
from data_preprocessing import create_data_generators
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def get_num_classes(train_dir):
    train_generator, _ = create_data_generators(train_dir, None, batch_size=1)
    return len(train_generator.class_indices)

def train_model(train_dir, validation_dir):
    num_classes = get_num_classes(train_dir)
    train_generator, validation_generator = create_data_generators(train_dir, validation_dir, batch_size=32)
    
    model = build_model(num_classes)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
    
    model.fit(
        train_generator,
        epochs=50,  # Aumentamos el número de epochs
        validation_data=validation_generator,
        callbacks=[early_stopping, checkpoint]
    )
    
    return model

if __name__ == "__main__":
    train_dir = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/images/train'
    validation_dir = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/images/validation'
    
    model = train_model(train_dir, validation_dir)
