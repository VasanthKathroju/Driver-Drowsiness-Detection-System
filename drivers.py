from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pygame.mixer
from scipy.spatial import distance
import time

# ---------------- FLASK ----------------
app = Flask(__name__)

# ---------------- AUDIO -------------- --
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("beep-warning-1120.mp3")

# ---------------- MEDIAPIPE ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- SHARED STATE ----------------
current_status = "INACTIVE"
blink_count = 0
closed_count = 0
alert_playing = False

sleep = drowsy = active = 0

EYE_AR_SLEEP = 0.20
EYE_AR_DROWSY = 0.25
CONSEC_FRAMES = 6

# ---------------- EAR FUNCTION ----------------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ---------------- VIDEO STREAM ----------------
def generate_frames():
    global current_status, blink_count, closed_count
    global sleep, drowsy, active, alert_playing

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                left_eye = np.array([(int(face_landmarks.landmark[i].x * w),
                                      int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE])
                right_eye = np.array([(int(face_landmarks.landmark[i].x * w),
                                       int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE])

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

                if ear < EYE_AR_SLEEP:
                    sleep += 1
                    drowsy = active = 0
                    if sleep >= CONSEC_FRAMES:
                        current_status = "SLEEP"
                        closed_count += 1
                        if not alert_playing:
                            alert_sound.play()
                            alert_playing = True

                elif ear < EYE_AR_DROWSY:
                    drowsy += 1
                    sleep = active = 0
                    if drowsy >= CONSEC_FRAMES:
                        current_status = "DROWSY"
                        if not alert_playing:
                            alert_sound.play()
                            alert_playing = True

                else:
                    active += 1
                    sleep = drowsy = 0
                    if active >= CONSEC_FRAMES:
                        current_status = "ACTIVE"
                        blink_count += 1
                        alert_playing = False

                cv2.putText(frame, current_status, (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                            (0, 255, 0) if current_status == "ACTIVE" else
                            (0, 255, 255) if current_status == "DROWSY" else
                            (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({
        "status": current_status,
        "blinks": blink_count,
        "closed": closed_count
    })

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)