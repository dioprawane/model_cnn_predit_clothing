import os
import shutil
import random

# Chemin vers le dossier principal où se trouvent les dossiers bottomwear, topwear, shoes
base_folder = 'all'

# Chemin des dossiers de sortie
train_folder = 'data/train'
test_folder = 'data/test'
val_folder = 'data/validation'

# Proportions pour la division des données
train_ratio = 0.75
test_ratio = 0.2
val_ratio = 0.5

def create_folders(base_folder, category, train_folder, test_folder, val_folder):
    """
    Créer les dossiers train, test, validation pour chaque catégorie.
    """
    os.makedirs(os.path.join(train_folder, category), exist_ok=True)
    os.makedirs(os.path.join(test_folder, category), exist_ok=True)
    os.makedirs(os.path.join(val_folder, category), exist_ok=True)

def split_data(category_folder, train_folder, test_folder, val_folder):
    """
    Fonction pour séparer les images en ensembles d'entraînement, test et validation.
    """
    # Lister les fichiers dans le dossier de la catégorie
    images = os.listdir(category_folder)
    random.shuffle(images)  # Mélanger les images aléatoirement
    
    # Calculer les tailles pour chaque ensemble
    train_size = int(train_ratio * len(images))
    test_size = int(test_ratio * len(images))
    
    # Diviser les fichiers en train, test, et validation
    train_files = images[:train_size]
    test_files = images[train_size:train_size + test_size]
    val_files = images[train_size + test_size:]
    
    # Copier les fichiers dans les bons dossiers
    for file in train_files:
        shutil.copy(os.path.join(category_folder, file), os.path.join(train_folder, file))
    
    for file in test_files:
        shutil.copy(os.path.join(category_folder, file), os.path.join(test_folder, file))
    
    for file in val_files:
        shutil.copy(os.path.join(category_folder, file), os.path.join(val_folder, file))
    
    # Affichage du nombre d'images pour chaque partie
    print(f"Catégorie {os.path.basename(category_folder)} :")
    print(f"  {len(train_files)} images dans train")
    print(f"  {len(test_files)} images dans test")
    print(f"  {len(val_files)} images dans validation")
    
    # Retourner les nombres d'images pour chaque ensemble
    return len(train_files), len(test_files), len(val_files)

# Liste des catégories (dossiers)
categories = ['bottomwear', 'topwear', 'shoes']

# Créer les sous-dossiers train, test, validation pour chaque catégorie
for category in categories:
    create_folders(base_folder, category, train_folder, test_folder, val_folder)

# Stocker les totaux d'images par catégorie et par ensemble
total_images = {
    "train": 0,
    "test": 0,
    "validation": 0
}

# Séparer les données pour chaque catégorie et afficher les statistiques
for category in categories:
    category_folder = os.path.join(base_folder, category)
    train_count, test_count, val_count = split_data(
        category_folder, 
        os.path.join(train_folder, category), 
        os.path.join(test_folder, category), 
        os.path.join(val_folder, category)
    )
    
    # Additionner le total d'images pour chaque partie (train, test, validation)
    total_images["train"] += train_count
    total_images["test"] += test_count
    total_images["validation"] += val_count

# Affichage du nombre total d'images dans chaque partie
print(f"\nNombre total d'images dans chaque ensemble :")
print(f"  Train: {total_images['train']} images")
print(f"  Test: {total_images['test']} images")
print(f"  Validation: {total_images['validation']} images")
