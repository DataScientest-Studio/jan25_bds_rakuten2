# === IMPORTS DES LIBRAIRIES ===
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import numpy as np
import os
import re
import html
import pandas as pd
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
sorted_labels = sorted(labelcat.keys())
index_to_label = {i: sorted_labels[i] for i in range(len(sorted_labels))}

# === UTILS ===
def label_to_string(label_id):
    return labelcat.get(label_id, f"Catégorie inconnue ({label_id})")

def preprocess_text(text):
    """Nettoyage du texte HTML, suppression des caractères spéciaux, normalisation."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = html.unescape(text)
    text = re.sub(r"[^a-zA-Z0-9\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]", " ", text)
    return re.sub(r'\s+', ' ', text).strip()

# === CHARGEMENT DU MODÈLE CAMEMBERT ===
def load_text_model():
    # A COMPLÉTER : adapter le chemin changesment de modèle ou d’organisation
    model_path = r"C:\\Users\\rudy_\\Documents\\Datascientest\\Projet Rakuten\\FINALE\\Camembert\\camembert_modele_1"
    model = CamembertForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = CamembertTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

# === PRÉDICTION MANUELLE ===
def predict_camembert(text_input):
    model, tokenizer = load_text_model()
    text = preprocess_text(text_input)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze()
        pred_index = torch.argmax(probs).item()
    return index_to_label[pred_index], probs.cpu().numpy()

# === PRÉDICTION SUR DATAFRAME ===
def predict_dataframe(df, designation_col="designation", description_col="description", text_col=None, true_label_col="prdtypecode"):
    model, tokenizer = load_text_model()
    preds, y_true = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CamemBERT - Prédictions"):
        # Si text_col est précisé (ex: "txt_fr"), on l’utilise, sinon on concatène designation + description
        if text_col:
            text_input = str(row[text_col]).strip()
        else:
            text_input = f"{row.get(designation_col, '')} {row.get(description_col, '')}"
        text_input = preprocess_text(text_input)

        # Tokenisation + prédiction
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).squeeze()
            pred_index = torch.argmax(probs).item()
            preds.append(index_to_label[pred_index])
            y_true.append(row.get(true_label_col, None))

    df["predicted_label"] = preds
    df["predicted_category"] = df["predicted_label"].apply(label_to_string)

    # Évaluation uniquement sur les lignes valides
    y_clean = [(yt, yp) for yt, yp in zip(y_true, preds) if yt is not None and yp is not None]
    if y_clean:
        yt, yp = zip(*y_clean)
        report = classification_report(yt, yp, target_names=all_label_names, labels=all_labels, zero_division=0)
        print(report)
        with open("classification_report_camembert.txt", "w", encoding="utf-8") as f:
            f.write(report)

        # Matrice de confusion
        cm = confusion_matrix(yt, yp, labels=all_labels)
        plt.figure(figsize=(16, 12))
        sns.heatmap(cm, xticklabels=all_label_names, yticklabels=all_label_names, annot=True, fmt="d", cmap="Blues")
        plt.title("Matrice de confusion - CamemBERT")
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.tight_layout()
        plt.savefig("confusion_matrix_camembert.png")
        plt.close()

    return df

# === MAIN POUR TESTER ===
if __name__ == "__main__":
    try:
        text_input = "chaise pliante pour jardin en métal noir"  # A COMPLÉTER avec un autre exemple si souhaité
        pred, _ = predict_camembert(text_input)
        print(f"Prédiction manuelle : {pred} → {label_to_string(pred)}")
    except Exception as e:
        print("Erreur prédiction manuelle :", e)

    try:
        csv_path = r"C:\\Users\\rudy_\\Documents\\Datascientest\\Projet Rakuten\\FINALE\\Preprocessing\\dfs\\X_test_translated_BERT.csv"  # A COMPLÉTER si test sur un autre jeu
        df_test = pd.read_csv(csv_path)
        df_results = predict_dataframe(df_test, text_col="txt_fr")  # OU text_col=None si utilisation designation+description
        df_results.to_csv("resultats_predictions_camembert.csv", index=False)
        print("Fichier 'resultats_predictions_camembert.csv' généré.")
        print(df_results[["productid", "predicted_label", "predicted_category"]].head())
    except Exception as e:
        print("Erreur prédiction DataFrame :", e)
