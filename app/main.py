from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import os
import tempfile
import naivebayes
import mochila
import backprop_module as backprop

from mobilenet_module import classify_image

from clustering import run_clustering
from sentiment import predict_sentiment, init_sentiment_model



app = FastAPI(title="Algoritmos Inteligentes")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Pre-carga los modelos necesarios."""
    init_sentiment_model()

@app.get("/")
def index():
    return FileResponse("app/static/index.html")

class ClusteringSilhouette(BaseModel):
    kmeans: float
    dbscan: float
    agglo: float

class ClusteringLabels(BaseModel):
    kmeans: List[int]
    dbscan: List[int]
    agglo: List[int]

class ClusteringResponse(BaseModel):
    silhouette: ClusteringSilhouette
    labels: ClusteringLabels
    pca: List[List[float]]

class CancerImageResponse(BaseModel):
    diagnosis: str
    confidence: float

class Objeto(BaseModel):
    nombre: str
    peso: float
    valor: float

class MochilaRequest(BaseModel):
    capacidad: float
    objetos: List[Objeto]

class MochilaResponse(BaseModel):
    seleccion: List[Objeto]
    peso: float
    valor: float
    historia: List[int]

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str

class MobileNetResponse(BaseModel):
    label: str
    probability: float

class BackpropRequest(BaseModel):
    inputs: List[List[float]]
    outputs: List[List[float]]

class BackpropResponse(BaseModel):
    result: List[List[float]]


nb_state = {"model": None, "le": None, "features": None}

@app.post("/api/clustering", response_model=ClusteringResponse)
async def clustering_endpoint(file: UploadFile = File(...)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado, se espera un CSV")
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = run_clustering(tmp_path)
    finally:
        os.remove(tmp_path)

    return result

@app.post("/api/cancer-image", response_model=CancerImageResponse)
async def cancer_image_endpoint(file: UploadFile = File(...)):

    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(status_code=400, detail="Formato de imagen no soportado")

    data = await file.read()
    suffix = os.path.splitext(file.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        label, confidence = naivebayes.classify_cancer_image(tmp_path)
        return CancerImageResponse(diagnosis=label, confidence=confidence)
    finally:
        os.remove(tmp_path)

@app.post("/api/sentiment", response_model=SentimentResponse)
async def sentiment_endpoint(req: SentimentRequest):
    resultado = predict_sentiment(req.text)
    return SentimentResponse(sentiment=resultado)

@app.post("/api/mochila", response_model=MochilaResponse)
def ejecutar_mochila(req: MochilaRequest):
    mochila.objetos = [o.dict() for o in req.objetos]
    mochila.CAPACIDAD_MAXIMA = req.capacidad
    sel, peso, val, hist = mochila.ejecutar_genetico()
    return MochilaResponse(seleccion=sel, peso=peso, valor=val, historia=hist)

@app.post("/api/backprop", response_model=BackpropResponse)
def ejecutar_backprop(req: BackpropRequest):
    X = np.array(req.inputs)
    Y = np.array(req.outputs)
    out = backprop.ejecutar_backprop(
        inputs=X,
        expected_output=Y,
        epochs=10000,
        lr=0.1,
        hidden_neurons=2,
    )
    return BackpropResponse(result=np.round(out, 4).tolist())


@app.post("/api/mobilenet", response_model=MobileNetResponse)
async def ejecutar_mobilenet_api(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(400, "Formato de imagen no soportado")
    data = await file.read()
    suffix = os.path.splitext(file.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        label, prob = classify_image(tmp_path)
    finally:
        os.remove(tmp_path)
    return MobileNetResponse(label=label, probability=prob)