# sentiment.py
import threading
from typing import Optional
from fastapi import HTTPException
from transformers import pipeline, Pipeline

# Nombre del modelo de Transformers
MODEL_NAME = "pysentimiento/robertuito-sentiment-analysis"

# Analizador (se cargará en startup)
sentiment_analyzer: Optional[Pipeline] = None
_analyzer_lock = threading.Lock()


def init_sentiment_model() -> None:
    """
    Inicializa el pipeline de análisis de sentimiento de forma segura.
    """
    global sentiment_analyzer
    with _analyzer_lock:
        if sentiment_analyzer is None:
            try:
                sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=MODEL_NAME,
                    tokenizer=MODEL_NAME,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Error al cargar el modelo de sentimiento '{MODEL_NAME}': {e}"
                )


def predict_sentiment(texto: str) -> str:
    """
    Devuelve la clase de sentimiento ('POS', 'NEU', 'NEG') para el texto dado.
    Asegura que el modelo esté inicializado antes de usarlo.
    """
    if not texto or not isinstance(texto, str):
        raise HTTPException(status_code=400, detail="Texto inválido para análisis de sentimiento.")

    # Asegurar inicialización
    if sentiment_analyzer is None:
        init_sentiment_model()

    try:
        resultado = sentiment_analyzer(texto)[0]
        label = resultado.get('label')
        return label
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al predecir sentimiento: {e}"
        )
