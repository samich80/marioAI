import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os

image_size = (352, 40)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/",
    validation_split = 0.10,
    image_size = image_size,
    subset="training",
    seed=42690,
)

validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/",
    validation_split = 0.10,
    image_size = image_size,
    subset="validation",
    seed=42690,
)

class_names = train_ds.class_names

print (class_names)

normalization_layer = layers.experimental.preprocessing.Rescaling(1.0/255.0)
normalized_ds = train_ds.map(lambda x,y: (normalization_layer(x), y))
images, labels = next(iter(normalized_ds))

class_count = len(class_names)

checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Создаем колбек контрольной точки
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=5)


print (class_count)

cnn_model = Sequential([
      layers.experimental.preprocessing.Rescaling(1.0/255, input_shape = (352, 40, 3)),
      
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),

      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),

      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),

      layers.Conv2D(128, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),

      layers.Conv2D(256, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
 

      layers.Flatten(),

      layers.Dense(1024, activation='relu'),
      layers.Dense(class_count)
])

cnn_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)


epoch_count = 20

cnn_model.fit(train_ds, validation_data=validation_ds, epochs=epoch_count, callbacks = [cp_callback])

cnn_model.summary()

checkpoint_path = "model/cnn_model.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

cnn_model.save(checkpoint_path)