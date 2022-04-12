import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#from google.colab.patches import cv2_imshow
#import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from PIL import Image, ImageOps
import streamlit as st
#
st.set_option('deprecation.showfileUploaderEncoding', False)

model = tf.keras.models.load_model('weights_emotions3.haf5')

st.write("""
          # Facial Emotion Recognition
"""
)

st.write("This is a simple image classification web app to predict rosts")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

@st.cache(persist = True)
def import_and_predict(image_data, model):
    size = (48, 48)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)

    img_reshape = image[np.newaxis,...]

    prediction = model.predict(img_reshape)
        
    return prediction

#@st.cache(persist = True)
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("Happy")
    elif np.argmax(prediction) == 1:
        st.write("Neutral")
    elif np.argmax(prediction) == 3:
        st.write('Sad')

    st.text("Probability (0: Happy, 1: Neutral, 2: Sad)")
    st.write(prediction)