from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
import torch
import time

app = Flask(__name__)

# GPU check (
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load model
model = YOLO('yolov8n.pt')

current_fps = 0

# Camera setup
cap = None
for i in range(3):
    print(f"Trying camera index {i}...")
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} opened successfully!")
        break
    cap.release()

if cap is None or not cap.isOpened():
    print("No camera found!")
    exit()

def generate_frames():
    global current_fps 
    prev_time = time.time()  # Get the "previous" time
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Run detection
        results = model(frame, verbose=False)
        
        # Draw bounding boxes
        annotated_frame = results[0].plot()
        
        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        fps = round(fps, 1)
        prev_time = curr_time
        current_fps = fps # Saves the fps as a global variable 
        
        # Convert frame to JPEG for browser
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        # Send frame to browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route for video stream
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for fps
@app.route('/get_fps')
def get_fps():
    return str(current_fps)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


