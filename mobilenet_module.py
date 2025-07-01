import os
from typing import Tuple, List
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def load_mobilenet_model(model_path: str = None) -> object:
    """
    Carga el modelo MobileNetV2 personalizado desde un archivo .h5.
    Si no se especifica ruta, busca en 'models/mobilenet_model.h5' junto al script.
    """
    if model_path is None:
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, 'models', 'mobilenet_model.h5')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo MobileNet en {model_path}. "
            "Coloca tu archivo .h5 en esa ruta."
        )
    return load_model(model_path)


def load_labels(labels_path: str = None) -> List[str]:
    """
    Carga lista de etiquetas desde un archivo de texto, una etiqueta por línea.
    """
    if labels_path is None:
        base_dir = os.path.dirname(__file__)
        labels_path = os.path.join(base_dir, 'models', 'labels.txt')
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de etiquetas en {labels_path}. "
            "Coloca tu 'labels.txt' con una etiqueta por línea."
        )
    with open(labels_path, 'r', encoding='latin-1') as f:
        return [line.strip() for line in f if line.strip()]


global_model = load_mobilenet_model()
try:
    global_labels = load_labels()
except FileNotFoundError:
    global_labels = [
        "surprise", "fear", "happy",
        "neutral", "sad", "angry", "disgust"
    ]


def classify_image(img_path: str) -> Tuple[str, float]:
    """
    Clasifica la imagen usando el modelo MobileNetV2 personalizado.
    Devuelve la etiqueta y la probabilidad asociada.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    preds_raw = global_model.predict(x)
    if preds_raw is None:
        raise RuntimeError("Falló la predicción: preds_raw es None")
    preds = preds_raw.flatten()

    idx = int(np.argmax(preds))
    label = global_labels[idx] if idx < len(global_labels) else str(idx)
    prob = float(preds[idx])
    return label, prob
