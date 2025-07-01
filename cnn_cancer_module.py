"""
Módulo para clasificar mamografías usando el modelo CBIS-DDSM-CNN.
Descargado de Hugging Face Hub (maiurilorenzo/CBIS-DDSM-CNN).
"""
import json
import os
import numpy as np
import cv2
import tensorflow as tf
from huggingface_hub import hf_hub_download

# Repositorio y variables globales
_REPO_ID = "maiurilorenzo/CBIS-DDSM-CNN"
_model = None
_preprocessing_info = None


def init_cnn_model():
    """
    Descarga y carga el modelo y la configuración de preprocesamiento.
    """
    global _model, _preprocessing_info
    if _model is None or _preprocessing_info is None:
        # Descarga de archivos
        model_path = hf_hub_download(repo_id=_REPO_ID, filename="CNN_model.h5")
        prep_path = hf_hub_download(repo_id=_REPO_ID, filename="preprocessing.json")
        # Carga del modelo Keras
        _model = tf.keras.models.load_model(model_path)
        # Carga de configuración JSON
        with open(prep_path, "r") as f:
            _preprocessing_info = json.load(f)


def classify_cancer_image(path: str):
    """
    Carga el modelo, procesa la imagen y clasifica como 'Cancer' o 'Normal'.
    Retorna la etiqueta y la probabilidad de cáncer.
    """
    init_cnn_model()
    # Leer imagen
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocesamiento
    size = tuple(_preprocessing_info.get("target_size", [50, 50]))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    scale = _preprocessing_info.get("scale", 255.0)
    img = img.astype(np.float32) / scale
    batch = np.expand_dims(img, axis=0)

    # Inferencia
    preds = _model.predict(batch)
    prob = float(preds[0][0])
    thresh = _preprocessing_info.get("threshold", 0.5)
    label = "Cancer" if prob >= thresh else "Normal"

    return label, prob