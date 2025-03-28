# === IMPORTS DES LIBRAIRIES ===
import torch
from torchvision import models, transforms
from PIL import Image, ImageEnhance
import numpy as np
import os
import pandas as pd
import re
import html
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup

# === DÉTECTION DE L'APPAREIL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")

# === DICTIONNAIRE DES CATÉGORIES RAKUTEN ===
labelcat = {
    10: "Livre occasion", 40: "Jeu vidéo, accessoire tech.", 50: "Accessoire Console",
    60: "Console de jeu", 1140: "Figurine", 1160: "Carte Collection", 1180: "Jeu Plateau",
    1280: "Jouet enfant, déguisement", 1281: "Jeu de société", 1300: "Jouet tech",
    1301: "Paire de chaussettes", 1302: "Jeu extérieur, vêtement", 1320: "Autour du bébé",
    1560: "Mobilier intérieur", 1920: "Chambre", 1940: "Cuisine", 2060: "Décoration intérieure",
    2220: "Animal", 2280: "Revues et journaux", 2403: "Magazines, livres et BDs",
    2462: "Jeu occasion", 2522: "Bureautique et papeterie", 2582: "Mobilier extérieur",
    2583: "Autour de la piscine", 2585: "Bricolage", 2705: "Livre neuf", 2905: "Jeu PC"
}
all_labels = list(labelcat.keys())
all_label_names = [labelcat[k] for k in all_labels]

# === UTILITAIRE : ID → Nom de catégorie ===
def label_to_string(label_id):
    return labelcat.get(label_id, f"Catégorie inconnue ({label_id})")

# === PRÉTRAITEMENT D’IMAGE ===
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    
    # Améliorations visuelles (netteté, luminosité, saturation)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.5)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.2)

    # Ajustement des canaux R/V/B
    img_array = np.array(image).astype(np.float32)
    img_array[:, :, 0] *= 0.9   # Rouge
    img_array[:, :, 1] *= 1.05  # Vert
    img_array[:, :, 2] *= 1.05  # Bleu
    image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    # Transformation standard ResNet
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# === CHARGEMENT DU MODÈLE RESNET ===
def load_image_model(num_classes):
    model_path = r"C:\\Users\\rudy_\\Documents\\Datascientest\\Projet Rakuten\\FINALE\\RESNET50\\resnet50_model2b_20250327_014445.pth"  # A COMPLÉTER si changement de modèle
    model = models.resnet50(weights=None)  # Aucun poids ImageNet, à ajuster si besoin
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# === PRÉDICTION UNIQUE SUR IMAGE ===
def get_image_proba(model, image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
    return torch.softmax(logits, dim=1).squeeze().cpu().numpy()

def predict_image(image_path):
    model = load_image_model(num_classes=27)  # A COMPLÉTER si le nombre de classes change
    image_tensor = preprocess_image(image_path)
    proba = get_image_proba(model, image_tensor)
    pred_index = np.argmax(proba)
    return all_labels[pred_index], proba

# === PRÉDICTION EN LOT (DATAFRAME) ===
def predict_dataframe(df, imageid_col="imageid", productid_col="productid",
                      true_label_col="prdtypecode", image_dir=None, text_col=None):
    model = load_image_model(num_classes=27)
    preds, y_true = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="ResNet50 - Prédictions"):
        image_path = os.path.join(image_dir, f"image_{row[imageid_col]}_product_{row[productid_col]}.jpg")
        if not os.path.exists(image_path):
            print(f"Image manquante : {image_path}")
            preds.append(None)
            y_true.append(None)
            continue
        try:
            image_tensor = preprocess_image(image_path)
            proba = get_image_proba(model, image_tensor)
            pred_index = np.argmax(proba)
            pred_label = all_labels[pred_index]
            preds.append(pred_label)
            y_true.append(row.get(true_label_col, None))
        except Exception as e:
            print(f"Erreur image {image_path} : {e}")
            preds.append(None)
            y_true.append(None)

    df["predicted_label"] = preds
    df["predicted_category"] = df["predicted_label"].apply(label_to_string)

    # === ÉVALUATION ===
    y_clean = [(yt, yp) for yt, yp in zip(y_true, preds) if yt is not None and yp is not None]
    if y_clean:
        yt, yp = zip(*y_clean)
        report = classification_report(yt, yp, target_names=all_label_names, labels=all_labels, zero_division=0)
        print(report)
        with open("classification_report_resnet.txt", "w", encoding="utf-8") as f:
            f.write(report)

        cm = confusion_matrix(yt, yp, labels=all_labels)
        plt.figure(figsize=(16, 12))
        sns.heatmap(cm, xticklabels=all_label_names, yticklabels=all_label_names, annot=True, fmt="d", cmap="Blues")
        plt.title("Matrice de confusion - ResNet50")
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.tight_layout()
        plt.savefig("confusion_matrix_resnet.png")
        plt.close()
    return df

# === MAIN POUR UTILISATION INTERACTIVE ===
if __name__ == "__main__":
    try:
        image_path = r"C:\\Users\\rudy_\\Documents\\Datascientest\\Projet Rakuten\\images\\images\\test\\image_959218573_product_231451737.jpg"  # A COMPLÉTER
        pred, _ = predict_image(image_path)
        print(f"Prédiction manuelle : {pred} → {label_to_string(pred)}")
    except Exception as e:
        print("Erreur prédiction image manuelle :", e)

    try:
        csv_path = r"C:\\Users\\rudy_\\Documents\\Datascientest\\Projet Rakuten\\FINALE\\Preprocessing\\dfs\\X_test_translated_BERT.csv"  # A COMPLÉTER
        image_dir = r"C:\\Users\\rudy_\\Documents\\Datascientest\\Projet Rakuten\\images\\images\\test"  # A COMPLÉTER
        df_test = pd.read_csv(csv_path)
        df_results = predict_dataframe(df_test, image_dir=image_dir, text_col="txt_fr")
        df_results.to_csv("resultats_predictions_resnet.csv", index=False)
        print("Fichier 'resultats_predictions_resnet.csv' généré.")
        print(df_results[["productid", "predicted_label", "predicted_category"]].head())
    except Exception as e:
        print("Erreur prédiction DataFrame :", e)
