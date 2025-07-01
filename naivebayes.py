import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from PIL import Image
import warnings

# Librerías para Hugging Face
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor

warnings.filterwarnings('ignore')

# --- Pipeline tabular existente ---
# (Aquí va todo el código original de entrenar_modelo, predecir_con_pipeline,
#  _load_digits_model y clasificar_imagen para dígitos.)

# --- Modelo de mamografías (Hugging Face) ---

_HF_MODEL_ID = "maiurilorenzo/CBIS-DDSM-CNN"
_hf_model = None
_hf_extractor = None


def init_hf_cancer_model():
    """
    Inicializa el feature extractor y el modelo de Hugging Face para mamografías.
    Se llama de forma perezosa en la primera clasificación.
    """
    global _hf_model, _hf_extractor
    if _hf_model is None:
        _hf_extractor = AutoImageProcessor.from_pretrained(_HF_MODEL_ID)
        _hf_model = AutoModelForImageClassification.from_pretrained(_HF_MODEL_ID)
        _hf_model.eval()


def classify_cancer_image(path: str):
    """
    Clasifica una imagen de mamografía en 'Benigno' o 'Maligno'.
    Devuelve label (str) y confidence (float).
    """
    init_hf_cancer_model()
    img = Image.open(path).convert("RGB")
    inputs = _hf_extractor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = _hf_model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    idx = int(np.argmax(probs))
    label = _hf_model.config.id2label[idx]
    confidence = float(probs[idx])
    return label, confidence