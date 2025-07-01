import zipfile
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models, optimizers

def ejecutar_mobilenet(path_zip, epochs, batch_size, learning_rate, *, save_path=None):
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
    num_classes = len(train_ds.class_names)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    base_model.trainable = False
    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    eval_loss, eval_acc = model.evaluate(val_ds)
    eval_metrics = {'loss': eval_loss, 'accuracy': eval_acc}

    if save_path:
        model.save(save_path)

    return history, eval_metrics


_emotion_model = None
_emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


def _load_emotion_model():
    global _emotion_model
    if _emotion_model is None:
        if os.path.exists("emotion_model.h5"):
            _emotion_model = tf.keras.models.load_model("emotion_model.h5")
        else:
            ejecutar_mobilenet("archive.zip", epochs=3, batch_size=32, learning_rate=1e-4, save_path="emotion_model.h5")
            _emotion_model = tf.keras.models.load_model("emotion_model.h5")
    return _emotion_model


def predecir_emocion(img_path):
    """Devuelve la emoción detectada en la imagen."""
    model = _load_emotion_model()
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    prob = float(np.max(preds))
    label = _emotion_labels[idx]
    return label, prob
