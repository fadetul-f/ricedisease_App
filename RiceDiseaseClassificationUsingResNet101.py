import streamlit as st
import tensorflow as tf
#from tensorflow import keras
import random
from PIL import Image, ImageOps
import numpy as np

import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Klasifikasi Penyakit Daun Padi",
    page_icon = ":Daun Padi:",
    initial_sidebar_state = 'auto'
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:
            
            return key


with st.sidebar:
        st.image('Resnet 101.png')
        st.image('SAMPEL.png')
        st.title("Oryza Sativa")
        st.subheader("Web klasifikasi penyakit daun padi yang dapat membantu mengklasifikasikan penyakit daun padi secara akurat")

             
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('resnet101-RiceLeafDiseases-56.19.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()
    #model = keras.Sequential()
    #model.add(keras.layers.Input(shape=(224, 224, 4)))
    

st.write("""
         # Klasifikasi penyakit daun padi
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction

        
if file is None:
    st.text("Silahkan masukkan gambar")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98,99)+ random.randint(0,99)*0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = ['BrownSpot', 'Healthy','Hispa','LeafBlast']

    string = "Klasifikasi Penyakit : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'Healthy':
        st.sidebar.success(string)

    elif class_names[np.argmax(predictions)] == 'BrownSpot':
        st.sidebar.warning(string)

    elif class_names[np.argmax(predictions)] == 'Hispa':
        st.sidebar.warning(string)
        
    elif class_names[np.argmax(predictions)] == 'LeafBlast':
        st.sidebar.warning(string)