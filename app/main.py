from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import numpy as np
import os
import tempfile


import naivebayes
import mochila
import backprop_module as backprop

try:
    import mobilenet
except Exception:  
    mobilenet = None

from clustering import run_clustering


app = FastAPI(title="Algoritmos Inteligentes")

# Serve frontend
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def index():
    return FileResponse("app/static/index.html")

# ---- Naive Bayes ----
class NBTrainRequest(BaseModel):
    test_size: float = 0.2
    random_state: int = 22
    imputer_strategy: str = "mean"

class NBTrainResponse(BaseModel):
    accuracy: float
    features: List[str]
    logs: List[str]

class NBPredictRequest(BaseModel):
    values: List[float]

class NBPredictResponse(BaseModel):
    diagnosis: str

class NBImageResponse(BaseModel):
    label: int
    probability: float

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

nb_state = {"model": None, "le": None, "features": None}

@app.post("/api/clustering", response_model=ClusteringResponse)
async def clustering_endpoint(file: UploadFile = File(...)):
    # Aceptamos solo CSV
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado, se espera un CSV")

    # Guardar temporalmente
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = run_clustering(tmp_path)
    finally:
        os.remove(tmp_path)

    return result

@app.post("/api/naive_bayes/train", response_model=NBTrainResponse)
def train_nb(req: NBTrainRequest):
    model, le, feats, acc, logs = naivebayes.entrenar_modelo(
        test_size=req.test_size,
        random_state=req.random_state,
        imputer_strategy=req.imputer_strategy,
    )
    nb_state.update({"model": model, "le": le, "features": feats})
    return NBTrainResponse(accuracy=acc, features=feats, logs=logs)

@app.post("/api/naive_bayes/predict", response_model=NBPredictResponse)
def predict_nb(req: NBPredictRequest):
    if nb_state["model"] is None:
        raise HTTPException(status_code=400, detail="Modelo no entrenado")
    arr = np.array(req.values).reshape(1, -1)
    pred = nb_state["model"].predict(arr)[0]
    diag = nb_state["le"].inverse_transform([pred])[0]
    return NBPredictResponse(diagnosis=diag)


@app.post("/api/naive_bayes/image", response_model=NBImageResponse)
async def nb_from_image(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(status_code=400, detail="Formato de imagen no soportado")
    data = await file.read()
    suffix = os.path.splitext(file.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        label, prob = naivebayes.clasificar_imagen(tmp_path)
    finally:
        os.remove(tmp_path)
    return NBImageResponse(label=label, probability=prob)

# ---- Mochila ----
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

@app.post("/api/mochila", response_model=MochilaResponse)
def ejecutar_mochila(req: MochilaRequest):
    mochila.objetos = [o.dict() for o in req.objetos]
    mochila.CAPACIDAD_MAXIMA = req.capacidad
    sel, peso, val, hist = mochila.ejecutar_genetico()
    return MochilaResponse(seleccion=sel, peso=peso, valor=val, historia=hist)

# ---- Backpropagation ----
class BackpropRequest(BaseModel):
    inputs: List[List[float]]
    outputs: List[List[float]]

class BackpropResponse(BaseModel):
    result: List[List[float]]

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

# ---- MobileNetV2 ----

class MobileNetResponse(BaseModel):
    label: str
    probability: float


@app.post("/api/mobilenet", response_model=MobileNetResponse)
async def ejecutar_mobilenet_api(file: UploadFile = File(...)):
    if mobilenet is None:
        raise HTTPException(status_code=500, detail="TensorFlow no disponible")
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(status_code=400, detail="Formato de imagen no soportado")
    data = await file.read()
    suffix = os.path.splitext(file.filename)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        label, prob = mobilenet.clasificar_imagen(tmp_path)
    finally:
        os.remove(tmp_path)
    return MobileNetResponse(label=label, probability=prob)