import tensorflow as tf

# Charger votre modèle Keras
model = tf.keras.models.load_model("mon_modele_bis.h5")

# Créer un convertisseur TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Enregistrer le modèle converti
with open("mon_modele_bis.tflite", "wb") as f:
    f.write(tflite_model)