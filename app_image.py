from linecache import checkcache
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import h5py

st.set_option('deprecation.showfileUploaderEncoding', False)

#criando 3 colunas 
#col1, col2, col3 = st.beta_columns(3)
#foto = Image.open('canaa.png')
#inserindo na coluna 2
#col2.image(foto, use_column_width=True)

st.title('Bem vindo ao classificador de emoções!')

instructions = """
        Click no botão Browse files para carregar uma imagem. Após o carregamento da imagem, 
        click no botão predict e deixe que Rede Neural classifique
        a imagem em: Happy (feliz), Sad (triste) or Neutral (neutro)...
        """
st.write(instructions)

st.write("Esta é uma amostra de um simples classificador de emoões em imagens")

upload_file = st.file_uploader("Please upload an image file", type=['png', 'jpg'])

generate_pred = st.button("Predict")

model = tf.keras.models.load_model('model_emotions3.haf5')

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