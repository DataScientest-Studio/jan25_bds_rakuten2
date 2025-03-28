# === IMPORTS DES LIBRAIRIES ===
import torch
from torchvision import models, transforms
from PIL import Image, ImageEnhance
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import numpy as np
from joblib import load
import os
import re
import html
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, ClassifierMixin
from wrappers import CamembertWrapper, ResNetWrapper  # A COMPLÉTER : vérifier que ce fichier existe bien

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

# Mapping inverse : index numérique → code Rakuten
sorted_labels = sorted(labelcat.keys())
index_to_label = {i: sorted_labels[i] for i in range(len(sorted_labels))}

# === UTILITAIRE POUR CONVERTIR UN LABEL EN NOM LISIBLE ===
def label_to_string(label_id):
    return labelcat.get(label_id, f"Catégorie inconnue ({label_id})")

# === PRÉTRAITEMENT DU TEXTE ===
def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = html.unescape(text)
    text = re.sub(r"[^a-zA-Z0-9\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]", " ", text)
    return re.sub(r'\s+', ' ', text).strip()

# === PRÉTRAITEMENT DE L'IMAGE POUR RESNET ===
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.5)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.2)
    img_array = np.array(image).astype(np.float32)
    img_array[:, :, 0] *= 0.9
    img_array[:, :, 1] *= 1.05
    img_array[:, :, 2] *= 1.05
    image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# === CHARGEMENT DES MODÈLES ===
def load_text_model():
    model_path = r"C:\\Users\\rudy_\\Documents\\Datascientest\\Projet Rakuten\\FINALE\\Camembert\\camembert_modele_1" # A COMPLÉTER
    model = CamembertForSequenceClassification.from_pretrained(model_path)
    tokenizer = CamembertTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def load_image_model(num_classes):
    model_path = r"C:\\Users\\rudy_\\Documents\\Datascientest\\Projet Rakuten\\FINALE\\RESNET50\\resnet50_model2b_20250327_014445.pth" # A COMPLÉTER
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_voting_model():
    return load(r"C:\\Users\\rudy_\\Documents\\Datascientest\\Projet Rakuten\\FINALE\\L_fusion\\Sauvegarde modeles\\voting_classifier.joblib") # A COMPLÉTER

# === PRÉDICTIONS POUR CHAQUE MODALITÉ ===
def get_text_proba(model, tokenizer, text):
    text = preprocess_text(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=1).squeeze().numpy()

def get_image_proba(model, image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
    return torch.softmax(logits, dim=1).squeeze().cpu().numpy()

# === PRÉDICTION MULTIMODALE (TEXTE + IMAGE) ===
def predict_multimodal(text_input, image_path):
    text_model, tokenizer = load_text_model()
    image_model = load_image_model(num_classes=27)
    voting_model = load_voting_model()

    text_proba = get_text_proba(text_model, tokenizer, text_input)
    image_tensor = preprocess_image(image_path)
    image_proba = get_image_proba(image_model, image_tensor)

    combined = np.zeros((1, 1))  # Dummy X
    voting_model.estimators_[0].P_val = text_proba.reshape(1, -1)
    voting_model.estimators_[1].P_val = image_proba.reshape(1, -1)

    pred_index = voting_model.predict(combined)[0]
    return index_to_label[pred_index]

# === PRÉDICTION SUR UN DATAFRAME COMPLET ===
def predict_dataframe(df, designation_col="designation", description_col="description",
                      imageid_col="imageid", productid_col="productid",
                      true_label_col="prdtypecode", image_dir=None, text_col=None):
    text_model, tokenizer = load_text_model()
    image_model = load_image_model(num_classes=27)
    voting_model = load_voting_model()

    preds, y_true = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Prédictions"):
        text_input = str(row[text_col]).strip() if text_col else f"{row.get(designation_col, '')} {row.get(description_col, '')}"
        image_path = os.path.join(image_dir, f"image_{row[imageid_col]}_product_{row[productid_col]}.jpg")
        if not os.path.exists(image_path):
            preds.append(None)
            y_true.append(None)
            continue
        try:
            text_proba = get_text_proba(text_model, tokenizer, text_input)
            image_tensor = preprocess_image(image_path)
            image_proba = get_image_proba(image_model, image_tensor)
            combined = np.zeros((1, 1))
            voting_model.estimators_[0].P_val = text_proba.reshape(1, -1)
            voting_model.estimators_[1].P_val = image_proba.reshape(1, -1)
            pred_index = voting_model.predict(combined)[0]
            pred_label = index_to_label[pred_index]
            preds.append(pred_label)
            y_true.append(row.get(true_label_col, None))
        except Exception as e:
            preds.append(None)
            y_true.append(None)

    df["predicted_label"] = preds
    df["predicted_category"] = df["predicted_label"].apply(label_to_string)

    # Évaluation uniquement sur les lignes valides
    y_clean = [(yt, yp) for yt, yp in zip(y_true, preds) if yt is not None and yp is not None]
    if y_clean:
        yt, yp = zip(*y_clean)
        report = classification_report(yt, yp, target_names=all_label_names, labels=all_labels, zero_division=0)
        print(report)
        with open("classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        cm = confusion_matrix(yt, yp, labels=all_labels)
        plt.figure(figsize=(16, 12))
        sns.heatmap(cm, xticklabels=all_label_names, yticklabels=all_label_names, annot=True, fmt="d", cmap="Blues")
        plt.title("Matrice de confusion")
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()

    return df

# === MAIN POUR TESTS LOCAUX ===
if __name__ == "__main__":
    try:
        text_input = "eau chlore"  # A COMPLÉTER
        image_path = r"C:\\Users\\rudy_\\Documents\\Datascientest\\Projet Rakuten\\images\\images\\test\\image_959218573_product_231451737.jpg" # A COMPLÉTER
        pred = predict_multimodal(text_input, image_path)
        print(f"Prédiction manuelle : {pred} → {label_to_string(pred)}")
    except Exception as e:
        print("Erreur prédiction manuelle :", e)

    try:
        csv_path = r"C:\\Users\\rudy_\\Documents\\Datascientest\\Projet Rakuten\\FINALE\\Preprocessing\\dfs\\X_test_translated_BERT.csv" # A COMPLÉTER
        image_dir = r"C:\\Users\\rudy_\\Documents\\Datascientest\\Projet Rakuten\\images\\images\\test" # A COMPLÉTER
        df_test = pd.read_csv(csv_path)
        df_results = predict_dataframe(df_test, text_col="txt_fr", image_dir=image_dir)  # OU text_col=None si pas de colonne traduite
        df_results.to_csv("resultats_predictions_voting.csv", index=False)
        print("Fichier 'resultats_predictions_voting.csv' généré.")
        print(df_results[["productid", "predicted_label", "predicted_category"]].head())
    except Exception as e:
        print("Erreur prédiction DataFrame :", e)
