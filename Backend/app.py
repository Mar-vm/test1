import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response
import base64
import time

app = Flask(__name__)

# Cargar el modelo entrenado
model = load_model("your_model.tflite")

# Inicializar MediaPipe y OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Mapeo de etiquetas de clases
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "a", "e", "i", "u", "o", "b", "c", "d", "f", "g", "h", "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "k", "q", "x", "z", "te amo", "mucho", "yo"]

# Generar frames para el video
def gen_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, image = cap.read()
        if not success:
            break
        
        # Convertir la imagen a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Procesar la imagen con MediaPipe
        results = hands.process(image_rgb)

        class_label = ""  # Predicción de la seña
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extraer las coordenadas x, y, z de cada punto clave
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

                # Convertir a numpy y redimensionar para el modelo
                keypoints = np.array(keypoints).reshape(1, -1)

                # Realizar la predicción
                prediction = model.predict(keypoints)
                class_index = np.argmax(prediction)
                class_label = class_names[class_index]  # Obtener la clase predicha

        # Añadir texto con la predicción
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f"Predicción: {class_label}", (10, 30), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Codificar la imagen como JPEG
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        # Convertir la imagen a base64
        frame_b64 = base64.b64encode(frame).decode('utf-8')

        # Enviar la imagen en formato base64 al frontend
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para capturar video y enviar frames
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)


