from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def build_model(num_classes):
    # Define input layer
    inputs = Input(shape=(150, 150, 3))
    
    # Base model
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    
    # Freeze more layers
    for layer in base_model.layers[:-8]:
        layer.trainable = False

    # Get the output from base model
    x = base_model.output
    
    # Apply GlobalAveragePooling2D
    x = GlobalAveragePooling2D()(x)
    
    # Flatten the output before adding Dense layers
    x = Flatten()(x)
    
    # Add Dense and Dropout layers with increased regularization
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.002))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # Additional Dropout layer
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model with a reduced learning rate
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
