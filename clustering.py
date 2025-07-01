import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calcula el coeficiente de silueta, devolviendo -1 si no es aplicable.
    """
    unique_labels = set(labels)
    if len(unique_labels) > 1 and len(unique_labels) < len(X):
        return silhouette_score(X, labels)
    return -1.0


def run_clustering(
    file_path: str,
    n_clusters: int = 4,
    eps: float = 1.5,
    min_samples: int = 10
) -> dict:
    """
    Ejecuta clustering sobre un CSV de clientes.
    Devuelve puntuaciones de silueta, etiquetas y componentes PCA.
    """
    df = pd.read_csv(file_path)

    if 'Categoria_Favorita' in df.columns:
        df = pd.get_dummies(df, columns=['Categoria_Favorita'])

    X_scaled = StandardScaler().fit_transform(df.values)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    km_labels = kmeans.fit_predict(X_scaled)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    db_labels = dbscan.fit_predict(X_scaled)

    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    ag_labels = agglo.fit_predict(X_scaled)

    sil_km = safe_silhouette(X_scaled, km_labels)
    sil_db = safe_silhouette(X_scaled, db_labels)
    sil_ag = safe_silhouette(X_scaled, ag_labels)

    return {
        "silhouette": {
            "kmeans": float(sil_km),
            "dbscan": float(sil_db),
            "agglo": float(sil_ag)
        },
        "labels": {
            "kmeans": km_labels.tolist(),
            "dbscan": db_labels.tolist(),
            "agglo": ag_labels.tolist()
        },
        "pca": X_pca.tolist()
    }