import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
import os
import numpy as np

image_size = (352, 40)

checkpoint_path = "model/cnn_model.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

class_names = ['BOX', 'BRICKS', 'BUSH', 'CLOUD', 'ENEMY1', 'FLOWER', 'HILL', 'MARIO', 'PIPE', 'SQUARES', 'TURTLE']

cnn_model = keras.models.load_model(checkpoint_path)
cnn_model.summary()

def classifyImage(image_path):
  image = keras.preprocessing.image.load_img(image_path, target_size=image_size)
  
  img_array = keras.preprocessing.image.img_to_array(image)
  img_array = tf.expand_dims(img_array, 0)
  predictions = cnn_model.predict(img_array)

  score = tf.nn.softmax(predictions)
  print(class_names[np.argmax(score)], 100 * np.max(score))

classifyImage('dataset\PIPE\screen45_cut14.png')