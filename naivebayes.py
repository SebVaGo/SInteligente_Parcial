import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Identificador del modelo Skin Cancer MobileNetV1 en Hugging Face
_HF_MODEL_ID = "nateraw/skin-cancer-mobilenet-v1"

# Variables globales para carga perezosa\ n_hf_processor = None
_hf_model = None


def classify_cancer_image(path: str):
    """
    Carga el modelo (si no está inicializado), procesa la imagen y devuelve
    la etiqueta ('BENIGN' o 'MALIGNANT') y la confianza.
    """
    global _hf_processor, _hf_model
    # Inicializar modelo y procesador si es la primera llamada
    if _hf_processor is None or _hf_model is None:
        _hf_processor = AutoImageProcessor.from_pretrained(_HF_MODEL_ID)
        _hf_model     = AutoModelForImageClassification.from_pretrained(_HF_MODEL_ID)
        _hf_model.eval()

    # Abrir y preparar imagen
    img = Image.open(path).convert("RGB")
    inputs = _hf_processor(images=img, return_tensors="pt")

    # Inferencia
    with torch.no_grad():
        outputs = _hf_model(**inputs)
    logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    label = _hf_model.config.id2label[idx]
    confidence = float(probs[idx])

    return label, confidence