Rakuten France Multimodal Product Data Classification
==============================

This repository gathers all the scripts, notebooks, and tools developed as part of a product classification project for the Rakuten marketplace. This work was carried out during the Data Scientist training program by DataScientest.

The main objective is to automatically predict the category (`prdtypecode`) of a Rakuten product, using its **textual features** (designation, description) and/or **visual features** (product image).

---

## Data

The data used in this project is available via the following Hugging Face link:

https://huggingface.co/datasets/Rudy-Mev/Data/blob/main/data.zip

This archive includes all resources necessary to reproduce the experiments:

- Raw images: Split into `train`, `val`, and `test` folders. Each image is named `image_<imageid>_product_<productid>.jpg`.
- Raw dataframes: Containing original fields like `designation`, `description`, `prdtypecode`, and `imageid`.
- Preprocessed dataframes for Machine Learning: Cleaned, lemmatized, and ready for vectorization (e.g., for TF-IDF + SVM).
- CamemBERT-ready dataframes: Lightly preprocessed texts (cleaned, translated) compatible with Hugging Face BERT tokenizers.

---

## Notebooks

Several Jupyter notebooks were developed for data preprocessing:

- `1-Exploratory-Data-Analysis-&-SPLIT-Text.ipynb`: EDA on texts and labels, then splitting into train, validation, and test sets.
- `2-Exploratory-Data-Analysis-Images.ipynb`: Image exploration and structure analysis.
- `3-Data-Preprocessing-Text.ipynb`: Text cleaning, translation, tokenization, stopword removal, lemmatization.
- `4-Data-Preprocessing-Images.ipynb`: Image enhancement, resizing, normalization.

Modeling and evaluation notebooks:

- `5-Modeling-Unimodal-Machine-Learning.ipynb`: Various classical machine learning approaches using different combinations of vectorization techniques (TF-IDF, FastText) and classifiers (Random Forest, Logistic Regression, LinearSVC).
- `6-Modeling-Unimodal-CAMEMBERT.ipynb`: Fine-tuning CamemBERT on textual data.
- `y-Modeling-Unimodal-RESNET50.ipynb`: Training ResNet50 on product images.
- `z-Modeling-Multimodal-Late-Fusion.ipynb`: Combining text and image models using Max Rule, Voting, and Stacking (best results with XGBoost).

---

## Model Demonstrations

Each model is accompanied by a demo notebook that can run manual predictions or batch inference

Available demonstrations:
- CamemBERT
- ResNet50
- TF-IDF + SVM
- Multimodal fusion (Max Rule, Voting, Stacking)

---

## Model Structure

Trained models are organized into subfolders based on modality:

Each folder typically contains:

- The trained model (`.joblib` for scikit-learn or `.pt` for PyTorch)
- A Python inference script (`*_predictor.py`)
- A demonstration notebook
- Inference results:
  - CSV file with predictions (`resultats_predictions_*.csv`)
  - Confusion matrix image (`confusion_matrix.png`)
  - Classification report (`classification_report.txt`)

Some models are hosted externally due to size constraints:

- CamemBERT fine-tuned model: https://huggingface.co/Rudy-Mev/camembert_model
- Multimodal Stacking (text + image with XGBoost): https://huggingface.co/Rudy-Mev/multimodal_stacking_xgb

---

## Environment and Dependencies

A `requirements.txt` file is provided at the root of the repository. It lists all required Python packages (e.g., scikit-learn, transformers, torch, spacy).

To replicate the environment and ensure compatibility, it is recommended to create a virtual environment and install dependencies with:

```bash
pip install -r requirements.txt
```



Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
