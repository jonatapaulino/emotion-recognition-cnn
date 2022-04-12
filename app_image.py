import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import h5py

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Facial Emotion Recognition')

st.write("This is a simple web app for image classification")

upload_file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])

generate_pred = st.sidebar.button("Predict")

model = tf.keras.models.load_model('weights_emotions3.haf5')

def import_n_pred(image_data, model):
    size = (48,48)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape = img[np.newaxis,...]
    pred = model.predict(reshape)
    return pred
    
if generate_pred:
    image = Image.open(upload_file)
    with st.beta_expander('image', expanded=True):
        st.image(image, use_column_width=True)
    pred = import_n_pred(image, model)
    labels = ['Happy' , 'Neutral', 'Sad']
    st.title("prediction of image is {}".format(labels[np.argmax(pred)]))


##==============================================================================================================
##==============================================================================================================
# import cv2
# import numpy as np
# #import matplotlib.pyplot as plt
# #import seaborn as sns
# #from google.colab.patches import cv2_imshow
# #import zipfile
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
# from PIL import Image, ImageOps
# import streamlit as st
# #
# st.set_option('deprecation.showfileUploaderEncoding', False)

# model = tf.keras.models.load_model('weights_emotions3.haf5')

# st.write("""
#           # Facial Emotion Recognition
# """
# )

# st.write("This is a simple image classification web app to predict rosts")

# file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

# @st.cache(persist = True)
# def import_and_predict(image_data, model):
#     size = (48, 48)    
#     image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
#     image = image.convert('RGB')
#     image = np.asarray(image)
#     image = (image.astype(np.float32) / 255.0)

#     img_reshape = image[np.newaxis,...]

#     prediction = model.predict(img_reshape)
        
#     return prediction

# #@st.cache(persist = True)
# if file is None:
#     st.text("You haven't uploaded an image file")
# else:
#     image = Image.open(file)
#     st.image(image, use_column_width=True)
#     prediction = import_and_predict(image, model)
    
#     if np.argmax(prediction) == 0:
#         st.write("Happy")
#     elif np.argmax(prediction) == 1:
#         st.write("Neutral")
#     elif np.argmax(prediction) == 3:
#         st.write('Sad')

#     st.text("Probability (0: Happy, 1: Neutral, 2: Sad)")
#     st.write(prediction)