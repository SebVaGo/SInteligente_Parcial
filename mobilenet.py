import zipfile
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models, optimizers

def ejecutar_mobilenet(path_zip, epochs, batch_size, learning_rate):
    """
    Descomprime el ZIP de imágenes y entrena un modelo de transferencia usando MobileNetV2.

    Args:
        path_zip (str): Ruta al archivo ZIP con carpetas de clases.
        epochs (int): Número de épocas de entrenamiento.
        batch_size (int): Tamaño de lote.
        learning_rate (float): Tasa de aprendizaje.

    Returns:
        history: Objeto History de Keras con métricas de entrenamiento.
        eval_metrics: Diccionario con 'loss' y 'accuracy' sobre el dataset de validación.
    """
    work_dir = os.path.splitext(path_zip)[0]
    if not os.path.isdir(work_dir):
        with zipfile.ZipFile(path_zip, 'r') as z:
            z.extractall(work_dir)

    # Preparar datasets
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        work_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(224, 224),
        batch_size=batch_size
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        work_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(224, 224),
        batch_size=batch_size
    )

    # Obtener número de clases
    num_classes = len(train_ds.class_names)

    # Prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # Cargar MobileNetV2 base
    base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    base_model.trainable = False

    # Cabeza personalizada
    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)

    # Compilar
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Entrenar
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Evaluar
    eval_loss, eval_acc = model.evaluate(val_ds)
    eval_metrics = {'loss': eval_loss, 'accuracy': eval_acc}

    return history, eval_metrics


_clf_model = None


def _load_model():
    global _clf_model
    if _clf_model is None:
        _clf_model = MobileNetV2(weights="imagenet")
    return _clf_model


def clasificar_imagen(img_path):
    """Clasifica una imagen usando MobileNetV2 preentrenada en ImageNet."""
    model = _load_model()
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    preds = model.predict(arr)
    label, prob = decode_predictions(preds, top=1)[0][0][1:]
    return label, float(prob)
