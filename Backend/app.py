import time
import cv2
import json
import numpy as np
from flask import Flask, Response
from tensorflow.keras.models import load_model
import mediapipe as mp

app = Flask(__name__)

# Cargar el modelo entrenado
model = load_model("modelo_reconocimiento_senas_1_tipo1_33.h5")

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Mapeo de etiquetas de clases 
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "a", "e", "i", "u", "o", "b", "c", "d", "f", "g", "h", "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "k", "q", "x", "z", "te amo", "mucho", "yo"]

# Función para predecir el signo de la mano
def predict_hand_sign(points):
    # Preprocesar los puntos para que coincidan con la entrada del modelo
    processed_points = np.array(points).reshape(1, -1)  # Asegúrate de que coincida con la entrada de tu modelo
    prediction = model.predict(processed_points)
    class_index = np.argmax(prediction)
    class_label = class_names[class_index]  # Obtener la clase predicha
    return class_label

@app.route('/hand_points')
def hand_points():
    def generate():
        cap = cv2.VideoCapture(0)  # Captura de la cámara
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("No se pudo capturar el video.")
                break

            # Convertir la imagen a RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Procesar la imagen con MediaPipe
            results = hands.process(image_rgb)

            # Lista para almacenar los puntos de la mano
            points = []
            class_label = ""  # Predicción de la seña
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extraer las coordenadas x, y, z de cada punto clave
                    for lm in hand_landmarks.landmark:
                        points.extend([lm.x, lm.y, lm.z])

                # Realizar la predicción
                class_label = predict_hand_sign(points)

            # Enviar los puntos y la predicción al frontend
            yield f"data: {json.dumps({'points': points, 'prediction': class_label})}\n\n"
            time.sleep(1)  # Esperar un segundo antes de procesar el siguiente frame

        cap.release()  # Cerrar la captura de video

    return Response(generate(), content_type='text/event-stream')


if __name__ == "__main__":
    app.run(debug=True)

