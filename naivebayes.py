import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from PIL import Image
import numpy as np

def entrenar_modelo(
    url: str = 'https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv',
    test_size: float = 0.2,
    random_state: int = 22,
    imputer_strategy: str = 'mean'
):
    logs = []
    cancer = pd.read_csv(url)
    logs.append(f"Cargado CSV con {cancer.shape[0]} filas y {cancer.shape[1]} columnas")

    # Codificar diagnóstico
    le = LabelEncoder()
    cancer['diagnosis'] = le.fit_transform(cancer['diagnosis'])
    logs.append(f"Codificado diagnóstico: {sorted(le.classes_)} → {[0,1]}")

    # Selección de features, eliminando columnas sin nombre útiles
    X = cancer.drop(columns=['id','diagnosis'])
    # Eliminamos cualquier columna 'Unnamed' que esté vacía
    unnamed = [c for c in X.columns if c.startswith('Unnamed')]
    if unnamed:
        X = X.drop(columns=unnamed)
        logs.append(f"Eliminadas columnas vacías: {unnamed}")

    y = cancer['diagnosis']
    logs.append(f"Features seleccionados: {len(X.columns)} columnas")

    # Imputación
    imputer = SimpleImputer(strategy=imputer_strategy)
    X_imp = imputer.fit_transform(X)
    logs.append(f"Imputación ({imputer_strategy}) completa; nueva forma: {X_imp.shape}")

    # Reconstruir DataFrame con los nombres correctos
    X = pd.DataFrame(X_imp, columns=X.columns)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logs.append(f"Split train/test: test_size={test_size}, random_state={random_state}")
    logs.append(f"  → Train: {X_train.shape}, Test: {X_test.shape}")

    # Entrenamiento
    model = GaussianNB()
    model.fit(X_train, y_train)
    logs.append("Modelo GaussianNB entrenado")

    # Precisión
    accuracy = model.score(X_test, y_test)
    logs.append(f"Precisión sobre test: {accuracy:.4f}")

    return model, le, X.columns.tolist(), accuracy, logs


# --- Clasificación de imágenes ----
_digits_model = None


def _load_digits_model():
    """Entrena un clasificador binario (cáncer/no) con GaussianNB."""
    global _digits_model
    if _digits_model is None:
        digits = load_digits()
        # Usamos un mapeo simple de dígitos a clases para este ejemplo:
        # 0-4 -> no cáncer (0), 5-9 -> cáncer (1)
        y = (digits.target >= 5).astype(int)
        X_train, X_test, y_train, _ = train_test_split(
            digits.data, y, test_size=0.2, random_state=42
        )
        model = GaussianNB()
        model.fit(X_train, y_train)
        _digits_model = model
    return _digits_model


def clasificar_imagen(path):
    """Clasifica una imagen como cáncer (1) o no cáncer (0)."""
    model = _load_digits_model()
    img = Image.open(path).convert("L").resize((8, 8))
    arr = np.array(img, dtype=float)
    arr = 16 * arr / 255.0
    arr = arr.reshape(1, -1)
    probs = model.predict_proba(arr)[0]
    pred = int(probs.argmax())
    prob = float(probs[pred])
    return pred, prob
