import os
import json
from data_preprocessing import create_data_generators
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        print(f"Epoch {epoch + 1} ended with accuracy: {acc:.4f} and validation accuracy: {val_acc:.4f}")

    def on_train_end(self, logs=None):
        val_accs = self.model.history.history['val_accuracy']
        best_val_acc = max(val_accs)
        print(f"Training finished. Best validation accuracy: {best_val_acc:.4f}")

def get_num_classes_and_save_mapping(train_dir, mapping_path):
    train_generator, _ = create_data_generators(train_dir, train_dir, batch_size=1)  # Usar train_dir para train y validation temporalmente
    class_indices = train_generator.class_indices
    num_classes = len(class_indices)

    # Save class mapping to a JSON file
    with open(mapping_path, 'w') as f:
        json.dump(class_indices, f)

    return num_classes

def train_model(train_dir, validation_dir, mapping_path):
    num_classes = get_num_classes_and_save_mapping(train_dir, mapping_path)
    train_generator, validation_generator = create_data_generators(train_dir, validation_dir, batch_size=16)
    
    model = build_model(num_classes)
    
<<<<<<< HEAD
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint('C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/best_model.keras', monitor='val_accuracy', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
=======
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
>>>>>>> parent of f29725c (Accuracy 18% con una epoca)
    custom_callback = CustomCallback()
    
    history = model.fit(
        train_generator,
<<<<<<< HEAD
        epochs=15,
=======
        epochs=50,
>>>>>>> parent of f29725c (Accuracy 18% con una epoca)
        validation_data=validation_generator,
        callbacks=[early_stopping, checkpoint, custom_callback]
    )
    
    return model

if __name__ == "__main__":
    train_dir = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/images/train'
    validation_dir = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/images/validation'
    mapping_path = 'C:/Users/monti001/Documents/Trabajos/Progra/ML-Animals/data/class_indices.json'
    
    model = train_model(train_dir, validation_dir, mapping_path)
