import os
import json
import shutil

# Chemins des dossiers
styles_folder = "C:\\Users\\Diop Serigne Rawane\\.cache\\kagglehub\\datasets\\paramaggarwal\\fashion-product-images-dataset\\versions\\1\\fashion-dataset\\styles"
images_folder = "C:\\Users\\Diop Serigne Rawane\\.cache\\kagglehub\\datasets\\paramaggarwal\\fashion-product-images-dataset\\versions\\1\\fashion-dataset\\images"

# Chemin de destination pour les images filtrées
output_folder = os.path.join(os.getcwd(), "all")

# Créer le dossier de destination s'il n'existe pas
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Styles désirés
desired_styles = ["Topwear", "Shoes", "Pants"]

# Dictionnaire pour compter les images par catégorie
category_count = {"Topwear": 0, "Shoes": 0, "Pants": 0}

# Parcourir les fichiers JSON dans le dossier styles
for style_file in os.listdir(styles_folder):
    style_path = os.path.join(styles_folder, style_file)
    
    # Lire le fichier JSON avec encodage UTF-8
    with open(style_path, 'r', encoding='utf-8') as f:
        style_data = json.load(f)
    
    # Vérifier si le produit est dans les styles désirés
    product_name = style_data.get('productDisplayName', "")
    for style in desired_styles:
        if style in product_name:
            # Copier l'image correspondante
            image_filename = style_file + ".jpg"  # Supposons que l'image a le même nom que le fichier JSON
            source_image_path = os.path.join(images_folder, image_filename)
            destination_image_path = os.path.join(output_folder, image_filename)
            
            # Copier l'image si elle existe
            if os.path.exists(source_image_path):
                shutil.copy(source_image_path, destination_image_path)
                category_count[style] += 1
            else:
                print(f"Image not found: {source_image_path}")

# Afficher le nombre d'images récupérées par catégorie
for category, count in category_count.items():
    print(f"Number of images for {category}: {count}")

print(f"Images filtered and copied to {output_folder}")