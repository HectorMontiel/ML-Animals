import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(hp, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(
        filters=hp.Int('filters1', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('kernel_size1', values=[3, 5]),
        activation='relu',
        input_shape=(150, 150, 3)
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(
        filters=hp.Int('filters2', min_value=64, max_value=256, step=64),
        kernel_size=hp.Choice('kernel_size2', values=[3, 5]),
        activation='relu'
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=hp.Int('units', min_value=128, max_value=512, step=128),
        activation='relu'
    ))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
