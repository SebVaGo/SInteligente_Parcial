# sentiment.py
import os
from typing import List
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from fastapi import HTTPException

# Configurar el analizador de sentimiento usando un modelo preentrenado
# Este modelo se descargará automáticamente la primera vez
MODEL_NAME = "pysentimiento/robertuito-sentiment-analysis"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    sentiment_analyzer = pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer
    )
except Exception as e:
    raise RuntimeError(
        f"Error al cargar el modelo de sentimiento '{MODEL_NAME}': {e}"
    )


def predict_sentiment(texto: str) -> str:
    """
    Devuelve la clase de sentimiento ('POS', 'NEU', 'NEG') para el texto dado.
    Usa un pipeline de Transformers que descarga el modelo automáticamente.
    """
    if not texto or not isinstance(texto, str):
        raise HTTPException(status_code=400, detail="Texto inválido para análisis de sentimiento.")
    try:
        resultado = sentiment_analyzer(texto)[0]
        # El modelo de pysentimiento usa etiquetas como 'POS', 'NEU', 'NEG'
        label = resultado.get('label')
        return label
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al predecir sentimiento: {e}"
        )