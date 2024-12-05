import os
import sys
import time
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
import mediapipe as mp
import cv2
import json

# Ruta al modelo
model_path = os.path.join(os.getcwd(), 'Backend', 'your_model.tflite')  # Ruta local cuando el script no está empaquetado

# Inicializar Flask
app = Flask(__name__)

# Intentar cargar el modelo de Keras y manejar el error si el formato es incorrecto
try:
    model = load_model(model_path, compile=False)
except ValueError as e:
    sys.exit(1)  # Terminar el script si el modelo no se puede cargar

# Inicializar MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Mapeo manual de las clases
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "a", "e", "i", "u", "o", "b", "c", "d", "f", "g", "h", 
               "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "k", "q", "x", "z", "te amo", "mucho", "yo"]

# Función para capturar el video y procesarlo
def generate_frames():
    cap = cv2.VideoCapture(0)  # Captura de la cámara local

    # Verificar si la cámara se abrió correctamente
    if not cap.isOpened():
        return jsonify({"error": "No se pudo acceder a la cámara."}), 500

    while True:
        success, image = cap.read()

        if not success or image is None:
            continue  # O terminar el ciclo si prefieres no seguir ejecutando

        # Convertir la imagen a RGB para MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Procesar la imagen con MediaPipe
        results = hands.process(image_rgb)

        keypoints = []  # Aquí almacenaremos los puntos clave de la mano

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

        # Enviar los puntos clave en formato SSE
        if keypoints:
            yield f"data: {json.dumps(keypoints)}\n\n"

        time.sleep(1)  # Actualiza los puntos cada segundo (ajusta según sea necesario)

# Ruta para mostrar el video en el navegador (streaming)
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ruta para enviar los puntos clave de la mano a través de SSE
@app.route('/hand_points')
def hand_points():
    def generate():
        while True:
            points = []  # Esta es la lista de puntos clave que enviaremos
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()

            # Verificar si la captura fue exitosa
            if not ret:
                continue  # Si no, ignoramos el ciclo actual y continuamos

            # Convertimos la imagen a RGB y la procesamos con MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Si encontramos puntos de la mano, los añadimos a la lista
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        points.extend([lm.x, lm.y, lm.z])

            # Si encontramos puntos de la mano, los enviamos
            if points:
                yield f"data: {json.dumps(points)}\n\n"
            
            time.sleep(1)  # Actualiza cada segundo (ajustable)

    return Response(generate(), content_type='text/event-stream')

# Ruta principal de la aplicación web
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
