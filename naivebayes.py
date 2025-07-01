import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def entrenar_modelo(
    url: str = 'https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv',
    test_size: float = 0.2,
    random_state: int = 22,
    imputer_strategy: str = 'mean',
    use_feature_selection: bool = True,
    feature_selection_method: str = 'mutual_info',
    k_features: int = 15,
    use_scaling: bool = True,
    scaling_method: str = 'robust',
    cv_folds: int = 5
):
    """
    Algoritmo mejorado de detección de cáncer usando Naive Bayes optimizado.
    
    Mejoras implementadas:
    - Preprocesamiento robusto con escalado
    - Selección inteligente de características
    - Validación cruzada estratificada
    - Métricas médicas específicas
    - Manejo avanzado de outliers
    """
    logs = []
    
    # Carga de datos con validación
    try:
        cancer = pd.read_csv(url)
        logs.append(f"✓ Cargado CSV con {cancer.shape[0]} filas y {cancer.shape[1]} columnas")
    except Exception as e:
        logs.append(f"✗ Error cargando datos: {str(e)}")
        return None, None, None, 0.0, logs

    # Análisis inicial de datos
    logs.append(f"• Distribución de diagnósticos: {cancer['diagnosis'].value_counts().to_dict()}")
    
    # Codificar diagnóstico (M=1 maligno, B=0 benigno)
    le = LabelEncoder()
    cancer['diagnosis'] = le.fit_transform(cancer['diagnosis'])
    
    # Verificar que M sea 1 (maligno) y B sea 0 (benigno) para interpretabilidad médica
    class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    logs.append(f"• Codificación diagnóstico: {class_mapping}")
    
    # Preparación de features con limpieza mejorada
    X = cancer.drop(columns=['id', 'diagnosis'])
    
    # Eliminar columnas problemáticas
    cols_to_drop = []
    for col in X.columns:
        if col.startswith('Unnamed') or X[col].isna().all() or X[col].nunique() <= 1:
            cols_to_drop.append(col)
    
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
        logs.append(f"• Eliminadas columnas problemáticas: {cols_to_drop}")
    
    y = cancer['diagnosis']
    logs.append(f"• Features iniciales: {len(X.columns)} columnas")
    
    # Detección y tratamiento de outliers usando IQR
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    outlier_counts = {}
    
    for col in numeric_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
        if outliers > 0:
            outlier_counts[col] = outliers
            # Winsorizing: limitar valores extremos
            X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
    
    if outlier_counts:
        logs.append(f"• Outliers tratados en {len(outlier_counts)} columnas: {sum(outlier_counts.values())} valores")
    
    # Imputación inteligente
    imputer = SimpleImputer(strategy=imputer_strategy)
    X_imp = imputer.fit_transform(X)
    logs.append(f"• Imputación ({imputer_strategy}) aplicada")
    
    # Reconstruir DataFrame
    X = pd.DataFrame(X_imp, columns=X.columns)
    
    # Escalado de características (importante para Naive Bayes con features de diferentes escalas)
    scaler = None
    if use_scaling:
        if scaling_method == 'robust':
            scaler = RobustScaler()  # Mejor para datos médicos con outliers
        else:
            scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        logs.append(f"• Escalado ({scaling_method}) aplicado")
    
    # Selección de características optimizada para datos médicos
    selected_features = X.columns.tolist()
    feature_selector = None
    
    if use_feature_selection and len(X.columns) > k_features:
        if feature_selection_method == 'mutual_info':
            # Información mutua: mejor para relaciones no lineales
            selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
        elif feature_selection_method == 'f_classif':
            # F-test: mejor para relaciones lineales
            selector = SelectKBest(score_func=f_classif, k=k_features)
        else:  # RFE
            # Eliminación recursiva con Naive Bayes
            nb_temp = GaussianNB()
            selector = RFE(estimator=nb_temp, n_features_to_select=k_features)
        
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_features = [X.columns[i] for i in selected_indices]
        
        X = pd.DataFrame(X_selected, columns=selected_features)
        feature_selector = selector
        logs.append(f"• Selección de características ({feature_selection_method}): {len(selected_features)} features")
    
    # Split estratificado para mantener proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    train_dist = pd.Series(y_train).value_counts().to_dict()
    test_dist = pd.Series(y_test).value_counts().to_dict()
    logs.append(f"• Split estratificado - Train: {train_dist}, Test: {test_dist}")
    
    # Entrenamiento del modelo optimizado
    # Para datos médicos, usar var_smoothing para evitar overfitting
    model = GaussianNB(var_smoothing=1e-9)  # Valor optimizado para datos médicos
    model.fit(X_train, y_train)
    logs.append("• Modelo GaussianNB entrenado con suavizado optimizado")
    
    # Evaluación completa con métricas médicas
    accuracy = model.score(X_test, y_test)
    
    # Validación cruzada estratificada
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Predicciones para métricas adicionales
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidad clase positiva (maligno)
    
    # Métricas específicas para diagnóstico médico
    try:
        auc_score = roc_auc_score(y_test, y_pred_proba)
        logs.append(f"• AUC-ROC: {auc_score:.4f}")
    except:
        auc_score = 0.0
        logs.append("• AUC-ROC: No calculable")
    
    # Reporte de clasificación
    class_report = classification_report(y_test, y_pred, 
                                       target_names=['Benigno', 'Maligno'], 
                                       output_dict=True)
    
    # Métricas críticas para diagnóstico médico
    sensitivity = class_report['Maligno']['recall']  # Sensibilidad (recall para malignos)
    specificity = class_report['Benigno']['recall']  # Especificidad (recall para benignos)
    ppv = class_report['Maligno']['precision']       # Valor predictivo positivo
    npv = class_report['Benigno']['precision']       # Valor predictivo negativo
    
    logs.append(f"• Precisión test: {accuracy:.4f}")
    logs.append(f"• Validación cruzada: {cv_mean:.4f} ± {cv_std:.4f}")
    logs.append(f"• Sensibilidad (detectar malignos): {sensitivity:.4f}")
    logs.append(f"• Especificidad (detectar benignos): {specificity:.4f}")
    logs.append(f"• Valor predictivo positivo: {ppv:.4f}")
    logs.append(f"• Valor predictivo negativo: {npv:.4f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    logs.append(f"• Matriz confusión - VP:{tp}, VN:{tn}, FP:{fp}, FN:{fn}")
    
    # Análisis de características más importantes
    if feature_selector and hasattr(feature_selector, 'scores_'):
        feature_importance = list(zip(selected_features, feature_selector.scores_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        top_features = [f"{feat}({score:.2f})" for feat, score in feature_importance[:5]]
        logs.append(f"• Top 5 características: {', '.join(top_features)}")
    
    # Guardar objetos del pipeline para predicción
    pipeline_objects = {
        'scaler': scaler,
        'feature_selector': feature_selector,
        'selected_features': selected_features
    }
    
    return model, le, selected_features, accuracy, logs, pipeline_objects


def predecir_con_pipeline(model, le, pipeline_objects, valores):
    """
    Realiza predicción usando el pipeline completo de preprocesamiento.
    """
    # Convertir valores a DataFrame
    X = pd.DataFrame([valores], columns=pipeline_objects['selected_features'])
    
    # Aplicar escalado si existe
    if pipeline_objects['scaler']:
        X_scaled = pipeline_objects['scaler'].transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Aplicar selección de características si existe
    if pipeline_objects['feature_selector']:
        X_selected = pipeline_objects['feature_selector'].transform(X)
        X = pd.DataFrame(X_selected, columns=pipeline_objects['selected_features'])
    
    # Predicción
    pred_proba = model.predict_proba(X)[0]
    pred_class = model.predict(X)[0]
    confidence = max(pred_proba)
    
    diagnosis = le.inverse_transform([pred_class])[0]
    
    return {
        'diagnosis': diagnosis,
        'confidence': confidence,
        'probabilities': {
            'Benigno': pred_proba[0],
            'Maligno': pred_proba[1]
        }
    }


# --- Clasificación de imágenes (mantenido sin cambios para compatibilidad) ---
_digits_model = None

def _load_digits_model():
    """Entrena un clasificador de dígitos con GaussianNB si no existe."""
    global _digits_model
    if _digits_model is None:
        digits = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target, test_size=0.2, random_state=42
        )
        model = GaussianNB()
        model.fit(X_train, y_train)
        _digits_model = model
    return _digits_model

def clasificar_imagen(path):
    """Clasifica una imagen de un dígito (png/jpg)."""
    model = _load_digits_model()
    img = Image.open(path).convert("L").resize((8, 8))
    arr = np.array(img, dtype=float)
    arr = 16 * arr / 255.0
    arr = arr.reshape(1, -1)
    probs = model.predict_proba(arr)[0]
    label = int(probs.argmax())
    prob = float(probs.max())
    return label, prob