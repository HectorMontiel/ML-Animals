import tensorflow as tf

# Verifica las GPUs disponibles
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
