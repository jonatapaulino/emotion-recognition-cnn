from linecache import checkcache
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import h5py

#st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Facial Emotion Recognition')

st.write("This is a simple web app for image classification")

upload_file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])

generate_pred = st.sidebar.button("Predict")

model = tf.keras.models.load_model('model_emotions3.haf5')

@checkcache
def import_pred(image_data, model):
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
    pred = import_pred(image, model)

    labels = ['Happy' , 'Neutral', 'Sad']
    st.title("Prediction of Image is {}".format(labels[np.argmax(pred)]))