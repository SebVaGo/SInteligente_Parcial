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

# Serve frontend
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Pre-carga los modelos necesarios."""
    init_sentiment_model()

@app.get("/")
def index():
    return FileResponse("app/static/index.html")

# ---- Naive Bayes ----
# ---- Naive Bayes Mejorado ----
class NBTrainRequest(BaseModel):
    test_size: float = 0.2
    random_state: int = 22
    imputer_strategy: str = "mean"
    # Nuevos parámetros para el algoritmo mejorado
    use_feature_selection: bool = True
    feature_selection_method: str = "mutual_info"  # "mutual_info", "f_classif", "rfe"
    k_features: int = 15
    use_scaling: bool = True
    scaling_method: str = "robust"  # "robust" o "standard"
    cv_folds: int = 5

class NBTrainResponse(BaseModel):
    accuracy: float
    features: List[str]
    logs: List[str]
    # Nuevas métricas médicas
    cv_accuracy: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    auc_score: Optional[float] = None

class NBPredictRequest(BaseModel):
    values: List[float]

class NBPredictResponse(BaseModel):
    diagnosis: str
    # Nuevos campos para predicción mejorada
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None

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

# Modelos Pydantic
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str


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
    """Entrenar modelo de detección de cáncer con algoritmo mejorado."""
    try:
        # Llamar al algoritmo mejorado
        result = naivebayes.entrenar_modelo(
            test_size=req.test_size,
            random_state=req.random_state,
            imputer_strategy=req.imputer_strategy,
            use_feature_selection=req.use_feature_selection,
            feature_selection_method=req.feature_selection_method,
            k_features=req.k_features,
            use_scaling=req.use_scaling,
            scaling_method=req.scaling_method,
            cv_folds=req.cv_folds
        )
        
        if len(result) == 6:
            # Nuevo formato con pipeline_objects
            model, le, feats, acc, logs, pipeline_objects = result
            nb_state.update({
                "model": model, 
                "le": le, 
                "features": feats,
                "pipeline_objects": pipeline_objects
            })
            
            # Extraer métricas adicionales de los logs
            cv_accuracy = None
            sensitivity = None
            specificity = None
            auc_score = None
            
            for log in logs:
                if "Validación cruzada:" in log:
                    cv_accuracy = float(log.split(":")[1].split("±")[0].strip())
                elif "Sensibilidad" in log:
                    sensitivity = float(log.split(":")[1].strip())
                elif "Especificidad" in log:
                    specificity = float(log.split(":")[1].strip())
                elif "AUC-ROC:" in log:
                    try:
                        auc_score = float(log.split(":")[1].strip())
                    except:
                        auc_score = None
            
            return NBTrainResponse(
                accuracy=acc, 
                features=feats, 
                logs=logs,
                cv_accuracy=cv_accuracy,
                sensitivity=sensitivity,
                specificity=specificity,
                auc_score=auc_score
            )
        else:
            # Formato anterior (compatibilidad)
            model, le, feats, acc, logs = result
            nb_state.update({"model": model, "le": le, "features": feats, "pipeline_objects": None})
            return NBTrainResponse(accuracy=acc, features=feats, logs=logs)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en entrenamiento: {str(e)}")

@app.post("/api/naive_bayes/predict", response_model=NBPredictResponse)
def predict_nb(req: NBPredictRequest):
    """Predicción mejorada usando el pipeline completo."""
    if nb_state["model"] is None:
        raise HTTPException(status_code=400, detail="Modelo no entrenado")
    
    try:
        # Si tenemos pipeline_objects, usar predicción mejorada
        if nb_state["pipeline_objects"] is not None:
            result = naivebayes.predecir_con_pipeline(
                nb_state["model"],
                nb_state["le"],
                nb_state["pipeline_objects"],
                req.values
            )
            return NBPredictResponse(
                diagnosis=result["diagnosis"],
                confidence=result["confidence"],
                probabilities=result["probabilities"]
            )
        else:
            # Predicción tradicional (compatibilidad)
            arr = np.array(req.values).reshape(1, -1)
            pred = nb_state["model"].predict(arr)[0]
            diag = nb_state["le"].inverse_transform([pred])[0]
            return NBPredictResponse(diagnosis=diag)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/api/naive_bayes/image", response_model=NBImageResponse)
async def nb_from_image(file: UploadFile = File(...)):
    """Clasificación de imágenes de dígitos (mantenido sin cambios)."""
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

# Endpoint en main.py
@app.post("/api/sentiment", response_model=SentimentResponse)
async def sentiment_endpoint(req: SentimentRequest):
    resultado = predict_sentiment(req.text)
    return SentimentResponse(sentiment=resultado)

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