import tensorflow as tf
from tensorflow import keras 
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model 

model = load_model('/Users/james/Downloads/Image_Classifier.keras')

st.header('Image Classifier Web Application')

data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

img_width = 180
img_height = 180

image = st.text_input('Enter image name: ', '/Users/james/Downloads/apple.jpg')

image_load = tf.keras.utils.load_img(image, target_size=(img_width, img_height))
imput_array = tf.keras.utils.img_to_array(image_load)
image_batch = tf.expand_dims(imput_array, 0)

predict = model.predict(image_batch)
score = tf.nn.softmax(predict)

st.image(image)
st.write("The image is " + data_cat[np.argmax(score)])
st.write("Accuracy is " + str(np.max(score)*100))
