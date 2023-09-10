#To make predictions:
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import initializers
import h5py
import numpy as np

max_features = 10000
sequence_length = 250
embedding_dim = 16

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,'[%s]' % re.escape(string.punctuation),'')

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label


vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

raw_model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)
  ])

raw_model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))



def load_weights(model,save_path):
    hf = h5py.File(save_path, 'r')
    for i in model.trainable_weights:
        res = hf.get(i.name)
        res = tf.convert_to_tensor(np.array(res))
        if res.shape == i.shape:
            i.assign(res)

save_path='trained_weights.h5'
load_weights(raw_model,save_path)

trained_model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string),
    vectorize_layer,
  raw_model,
  layers.Activation('sigmoid')
])

trained_model.compile(loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])

examples = ["Bad Movie Sucks"]

predictions=trained_model.predict(examples)
print(predictions[0][0])

# Low probability value (less than 0.5) means negaitve review
# High probability value (greater than 0.5) means positive review
