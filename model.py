import os
import shutil
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from sklearn.utils import class_weight

def checkFileIsGood(directory):
    """
    Vérifie si les fichiers dans un répertoire sont valides ou existent.
    :param directory: Le chemin vers le répertoire à vérifier.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Le répertoire {directory} n'existe pas.")
    
    # Vérifier si des fichiers existent dans le répertoire
    files = os.listdir(directory)
    if len(files) == 0:
        raise FileNotFoundError(f"Aucun fichier trouvé dans le répertoire {directory}.")
    
    # Tu peux ajouter d'autres vérifications comme le format des fichiers ici
    print(f"Tous les fichiers dans {directory} sont bons.")

def afficher_images(data_generator, n_images=5):
    """
    Affiche quelques images à partir du générateur de données.
    
    :param data_generator: Un générateur d'images (ImageDataGenerator)
    :param n_images: Nombre d'images à afficher
    """
    # Récupérer un lot d'images et leurs étiquettes
    images, labels = next(data_generator)

    # Configuration de la taille de la figure
    plt.figure(figsize=(15, 10))
    
    # Affichage des images
    for i in range(n_images):
        plt.subplot(1, n_images, i + 1)  # 1 ligne, n_images colonnes
        plt.imshow(images[i])  # Affiche directement les images couleur
        plt.title(f"Label: {np.argmax(labels[i])}")  # Afficher l'étiquette (classe)
        plt.axis('off')  # Masquer les axes
    
    plt.tight_layout()
    plt.show()

data_dir = "data/"
checkFileIsGood(data_dir)
#makeDirTrainAndValidation(data_dir)

# ------------------------------------
# Charger les données des datasets de training et de validation, les normaliser et les augmenter
# ------------------------------------
# Normalisation et augmentation des données
"""datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Chargement des données d'entraînement avec augmentation
train_data = datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=(64, 64),
    batch_size=100,
    class_mode='categorical',
    color_mode='rgb'  # Utilise les images en couleur (RGB)
)

# Chargement des données de validation (sans augmentation)
validation_data = datagen.flow_from_directory(
    os.path.join(data_dir, 'validation'),
    target_size=(64, 64),
    batch_size=100,
    class_mode='categorical',
    color_mode='rgb'  # Utilise les images en couleur (RGB)
)

# Chargement des données de test (sans augmentation)
test_data = datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=(64, 64),
    batch_size=100,
    class_mode='categorical',
    color_mode='rgb'  # Utilise les images en couleur (RGB)
)"""

# Normalisation des images
datagen = ImageDataGenerator(rescale=1./255)

# Chargement des données d'entraînement
train_data = datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Chargement des données de test (que tu vas utiliser pour la validation)
test_data = datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Chargement des données de validation
validation_data = datagen.flow_from_directory(
    os.path.join(data_dir, 'validation'),
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Vérifiez la cohérence des classes
print("Classes d'entraînement:", train_data.class_indices)
print("Classes de validation:", validation_data.class_indices)
print("Classes de test:", test_data.class_indices)

# ------------------------------------
# Affichage de quelques images
# ------------------------------------
"""afficher_images(train_data, n_images=5)
afficher_images(validation_data, n_images=5)
afficher_images(test_data, n_images=5)"""

# ------------------------------------
# Construction du modèle
# ------------------------------------

"""model = Sequential([
    Input(shape=(80, 80, 3)),  # Garder 3 canaux pour les images en couleur (RGB)
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_data.class_indices), activation='softmax')  # Ajustement pour le nombre de classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ------------------------------------
# Entraîner le modèle
# ------------------------------------

history = model.fit(train_data, validation_data=validation_data, epochs=15)"""

############################################################
# Construction du réseau de neurones
############################################################

model = Sequential()

# Construisez votre modèle ici

# Ajout de la première couche Conv2D de 96 neurones avec un kernet 3*3 et MaxPooling2D de 2*2
"""model.add(Input(shape=(64, 64, 3)))
# Ajouter une couche Reshape avant Conv2D
#model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))"""
# Utilisez ceci directement dans votre première couche Conv2D :
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Ajout d'une seconde couche Conv2D de 128 neurones et MaxPooling2D de 2*2
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Ajout d'une troisième couche Conv2D de 160 neurones et MaxPooling2D de 2*2
model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplatir les données avant d'entrer dans des couches Dense
model.add(Flatten())

# Ajout d'une couche dense de 1024 neurones
model.add(Dense(256, activation='relu'))

# Ajout d'une couche Dropout pour éviter le surapprentissage réduisant le taux à 0.25
model.add(Dropout(0.25))

# Ajout de la dernière couche de sortie
model.add(Dense(len(train_data.class_indices), activation='softmax'))

# Affichage du résumé du modèle
model.summary()

# Compilation du modèle
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

############################################################
# Apprentissage du modèle
############################################################

# Apprentissage du modèle
"""history = model.fit(
    train_data,  # Données d'entraînement (images + étiquettes)
    validation_data=validation_data,  # Données de validation (images + étiquettes)
    batch_size=128, 
    epochs=10,
    verbose=1
)"""
# Utilisation des données de test pour la validation pendant l'entraînement
history = model.fit(
    train_data,  # Utilisation des données d'entraînement
    validation_data=test_data,  # Utilisation des données de test comme validation
    epochs=10,
    verbose=1
)

# ------------------------------------
# Sauvegarder le modèle dans un fichier HDF5
# ------------------------------------
model.save('mon_modele_bis.h5')

############################################################
# Affichage des métriques sur l'ensemble de l'apprentissage
############################################################

fig = plt.figure()

plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['apprentissage : '+str(round(100*history.history['accuracy'][-1], 1))+"%", 
    'test : '+str(round(100*history.history['val_accuracy'][-1], 1))+"%"], 
    loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['apprentissage : '+str(round(history.history['loss'][-1], 1)), 
            'test : '+str(round(history.history['val_loss'][-1], 1))], 
    loc='lower right')

plt.tight_layout()
plt.show()

############################################################
# Génération des prédictions pour afficher les images mal classées
############################################################

predictions = model.predict(test_data)

y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

# Identification des indices des images mal classées
incorrect = np.where(y_pred != y_true)[0]

# Affichage de quelques images mal classées
num_images = 10
plt.figure(figsize=(10, 10))

# Utiliser next() pour obtenir un lot d'images à partir du générateur test_data
image_batch, _ = next(test_data)

for i, incorrect_idx in enumerate(incorrect[:num_images]):
    plt.subplot(1, num_images, i + 1)

    # Correction : Affichez directement l'image à partir du lot d'images récupéré
    plt.imshow(image_batch[incorrect_idx])  # Affiche l'image dans sa taille d'origine (64x64x3)

    predicted_label = y_pred[incorrect_idx]
    true_label = y_true[incorrect_idx]

    plt.title(f"Prédit: {predicted_label}\nVrai: {true_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()

############################################################
# Calcul et affichage de la matrice de confusion
############################################################

# Calcul de la matrice de confusion
cm = confusion_matrix(y_true, y_pred)
print("Matrice de confusion :")
print(cm)

def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues)
    plt.title("Matrice de confusion")
    plt.ylabel("Vraies classes")
    plt.xlabel("Classes prédites")
    plt.show()

plot_confusion_matrix(cm)