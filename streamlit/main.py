from pydoc import classname

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from markdown_it.rules_inline import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

#charge rle model
model=load_model("CNNfruits.h5")
#ls noms des classes
classnames=['apple','banana','orange']
#fonction de prediction
def predict(image):
    img = image.resize((32, 32))  # Assurez-vous que les dimensions sont correctes pour votre modèle
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)  # Renvoie un tableau de prédictions
    class_index = np.argmax(prediction)  # Utilisez 'prediction' ici
    confidence = np.max(prediction)  # Utilisez 'prediction' ici
    return classnames[class_index], confidence



#interface streamlit
st.title("Fruits Prediction")
st.write("ce modele va présire si un fruit est une banane ou une orange ou une pomme")
#chargement d'image
uploaded_file=st.file_uploader("charger l'image ",type=['png','jpg','jpeg'])
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption='image telechargée',use_column_width=True)
    #affichage
    with st.spinner("analyse en cours...."):
        classnames,confidence=predict(image)
        st.success(f"Résultats:{classnames} ({confidence*100:.2f}%)")