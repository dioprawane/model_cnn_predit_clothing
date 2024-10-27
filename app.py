import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# Définir les noms des classes du dataset Fashion MNIST
class_names = ['bottomwear', 'shoes', 'topwear']

# Charger votre modèle Keras entraîné pour MNIST
model_path = os.path.join(os.path.dirname(__file__), 'mon_modele.h5')
model = tf.keras.models.load_model(model_path)
#model = tf.keras.models.load_model('mon_modele.h5')

# Fonction pour prétraiter l'image téléchargée
def preprocess_image(image):
    size = (64, 64)  # Redimensionner l'image à la taille d'entrée du modèle
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)  # Redimensionner
    image = np.array(image) / 255.0  # Normaliser l'image
    if image.shape[-1] == 3:  # Si l'image est déjà en RGB
        image = image
    elif len(image.shape) == 2:  # Si l'image est en niveaux de gris (2 dimensions)
        image = np.stack([image]*3, axis=-1)  # Convertir en RGB en copiant les canaux
    image = image.reshape(1, 64, 64, 3)  # Ajouter les dimensions batch et canal
    return image

# Configuration de l'application Streamlit
st.title("DIOP Serigne Rawane :")

# Téléchargement de fichier dans Streamlit
uploaded_file = st.file_uploader("Choisissez une image de vêtement...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Ouvrir l'image téléchargée
    image = Image.open(uploaded_file)
    predicted_class = None

    # Afficher l'image téléchargée
    st.image(image, caption='Image téléchargée', use_column_width=True)
    st.write("")

    # Affichage de "Classification en cours"
    classification_status = st.empty()  # Utiliser `st.empty()` pour permettre une mise à jour dynamique
    classification_status.text("Classification en cours...")

    # Prétraiter l'image
    processed_image = preprocess_image(image)

    # Faire la prédiction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)

    # Mise à jour de "Classification terminée"
    classification_status.text("Classification terminée")

    # Afficher le label de l'image prédite
    predicted_label = class_names[predicted_class]
    st.write(f"Le modèle prédit que la catégorie de l'image est : **{predicted_label}**")