import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Charger le modèle pré-entraîné
model = load_model('mon_modele_bis.h5')

# Préparer les données de test
data_dir = "data/"
datagen = ImageDataGenerator(rescale=1./255)
test_data = datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=(64, 64),
    batch_size=1,  # Traiter une image à la fois pour l'affichage des erreurs
    class_mode='categorical',
    shuffle=False
)

# Obtenir les vraies étiquettes et les prédictions
true_labels = test_data.classes
pred_probs = model.predict(test_data)
pred_labels = np.argmax(pred_probs, axis=1)

# Dictionnaire pour obtenir les noms de classes à partir des indices
class_names = {v: k for k, v in test_data.class_indices.items()}

# Calcul de la matrice de confusion
cm = confusion_matrix(true_labels, pred_labels)

# Afficher la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_data.class_indices, yticklabels=test_data.class_indices)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Matrice de Confusion')
plt.show()

# Trouver des prédictions incorrectes
incorrect_indices = np.where(pred_labels != true_labels)[0]
print(f"Nombre d'éléments mal prédits : {len(incorrect_indices)}")

# Afficher 5 exemples de prédictions incorrectes
plt.figure(figsize=(6, 6))
for i, idx in enumerate(incorrect_indices[:6]):
    img, true_label_idx = test_data[idx][0], true_labels[idx]
    pred_label_idx = pred_labels[idx]

    true_label_name = class_names[true_label_idx]
    pred_label_name = class_names[pred_label_idx]
    
    plt.subplot(2, 3, i + 1)
    plt.imshow(img[0])  # Affiche l'image (format attendu)
    plt.title(f"Vrai: {true_label_idx} = {true_label_name}\nPrédit: {pred_label_idx} = {pred_label_name}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Afficher 5 exemples de prédictions incorrectes
plt.figure(figsize=(6, 6))
for i, idx in enumerate(incorrect_indices[6:12]):
    img, true_label_idx = test_data[idx][0], true_labels[idx]
    pred_label_idx = pred_labels[idx]

    true_label_name = class_names[true_label_idx]
    pred_label_name = class_names[pred_label_idx]
    
    plt.subplot(2, 3, i + 1)
    plt.imshow(img[0])  # Affiche l'image (format attendu)
    plt.title(f"Vrai: {true_label_idx} = {true_label_name}\nPrédit: {pred_label_idx} = {pred_label_name}")
    plt.axis('off')

plt.tight_layout()
plt.show()