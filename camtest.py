import cv2
from ultralytics import YOLO
import torch

# Checks the GPU model (I used an RTX 5060) and prints the model. 
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load model
model = YOLO('yolov8n.pt')

# Try different camera indexes
for i in range(3):
    print(f"Trying camera index {i}...")
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} opened successfully!")
        break
    cap.release()

if not cap.isOpened():
    print("No camera found!")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Run detection
    results = model(frame, verbose=False)
    
    # Draw bounding boxes
    annotated_frame = results[0].plot()
    
    # Show result
    cv2.imshow('Object Tracker', annotated_frame)
    
    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()