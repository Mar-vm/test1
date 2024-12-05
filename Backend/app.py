import os
import sys
import time
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, Response
from tensorflow.keras.models import load_model
import mediapipe as mp
import cv2
import cv2
import mediapipe as mp
from flask import Flask, jsonify, Response

app = Flask(__name__)

# Inicializa MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks(frame):
    # Convierte la imagen de BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesa la imagen y obtiene las landmarks de las manos
    result = hands.process(rgb_frame)
    
    landmarks = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                # Convierte las coordenadas en proporciones (0-1)
                landmarks.append((landmark.x, landmark.y, landmark.z))
                
    return landmarks

def generate_points():
    camera = cv2.VideoCapture(0)  # Inicia la cámara
    while True:
        ret, frame = camera.read()  # Captura un cuadro de video
        if not ret:
            break
        
        # Extrae los puntos de las manos usando MediaPipe
        landmarks = extract_hand_landmarks(frame)
        
        # Aquí puedes enviar los puntos al frontend como JSON
        yield f"data: {landmarks}\n\n"
        
    camera.release()

@app.route('/hand_points')
def hand_points():
    return Response(generate_points(), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
