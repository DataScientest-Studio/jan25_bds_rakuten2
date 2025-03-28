import os, re, html, string
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from joblib import load
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Dictionnaire des catégories Rakuten ===
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


def label_to_string(label_id):
    return labelcat.get(label_id, f"Catégorie inconnue ({label_id})")

# === Répertoire de sauvegarde des résultats
SAVE_DIR = r"C:\Users\rudy_\Documents\Datascientest\Projet Rakuten\FINALE\GITHUB\TFIDF SVM"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Chargement du modèle et vectorizer
MODEL_PATH = r"C:\Users\rudy_\Documents\Datascientest\Projet Rakuten\FINALE\Machine Learning\TF IDF\Sauvegarde\model_tfidf_txt_0_tok_stp_lem_maxfeat_10000_mindf_1_maxdf_085_LinearSVC.joblib"
VECTORIZER_PATH = r"C:\Users\rudy_\Documents\Datascientest\Projet Rakuten\FINALE\Machine Learning\TF IDF\Sauvegarde\vectorizer_tfidf_maxfeat_10000_mindf_1_maxdf_085.joblib"
model = load(MODEL_PATH)
vectorizer = load(VECTORIZER_PATH)

# === Prédiction individuelle
def predict_text(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

# === Prédiction sur un DataFrame (déjà prétraité)
def predict_dataframe(df, text_col="txt_0_tok_stp_lem", true_label_col="prdtypecode"):
    preds, y_true = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="TF-IDF - Prédictions"):
        text_input = str(row.get(text_col, ""))
        vec = vectorizer.transform([text_input])
        pred = model.predict(vec)[0]
        preds.append(pred)
        y_true.append(row.get(true_label_col, None))

    df["predicted_label"] = preds
    df["predicted_category"] = df["predicted_label"].apply(label_to_string)

    # Évaluation
    y_clean = [(yt, yp) for yt, yp in zip(y_true, preds) if yt is not None and yp is not None]
    if y_clean:
        yt, yp = zip(*y_clean)

        # Rapport de classification
        report = classification_report(
            yt, yp,
            labels=all_labels,
            target_names=all_label_names,
            zero_division=0
        )
        print(report)

        # Sauvegarde du rapport
        report_path = os.path.join(SAVE_DIR, "classification_report_tfidf.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Rapport de classification sauvegardé : {report_path}")

        # Matrice de confusion
        cm = confusion_matrix(yt, yp, labels=all_labels)
        plt.figure(figsize=(16, 12))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=all_label_names, yticklabels=all_label_names)
        plt.title("Matrice de confusion - TF-IDF")
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        cm_path = os.path.join(SAVE_DIR, "confusion_matrix_tfidf.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"Matrice de confusion sauvegardée : {cm_path}")

    return df

# === MAIN : prédiction manuelle + prédiction fichier
if __name__ == "__main__":
    # Prédiction sur une phrase libre
    example_text = "chaise pliante pour jardin en métal noir"
    try:
        pred = predict_text(example_text)
        print(f"\n[Manuel] Prédiction : {pred} → {label_to_string(pred)}")
    except Exception as e:
        print("Erreur prédiction manuelle :", e)

    # Prédiction sur DataFrame complet
    try:
        csv_path = r"C:\Users\rudy_\Documents\Datascientest\Projet Rakuten\FINALE\Preprocessing\dfs\X_test_fullpreprocessed.csv"
        df = pd.read_csv(csv_path)
        df_results = predict_dataframe(df, text_col="txt_0_tok_stp_lem")
        results_path = os.path.join(SAVE_DIR, "resultats_predictions_tfidf.csv")
        df_results.to_csv(results_path, index=False)
        print(f"Résultats sauvegardés dans : {results_path}")
        print(df_results[["predicted_label", "predicted_category"]].head())
    except Exception as e:
        print("Erreur prédiction DataFrame :", e)
