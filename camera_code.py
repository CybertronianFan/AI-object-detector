import cv2
from ultralytics import YOLO
import torch
import time


# Checks the GPU model (I used an RTX 5060) and prints the model
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

prev_time = time.time() # Get the "previous" time

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Run detection
    results = model(frame, verbose=False)
    
    # Draw bounding boxes
    annotated_frame = results[0].plot()

    #FPS calculation

    curr_time = time.time() # Get the current time
    fps = 1 / (curr_time - prev_time) # Calculates fps
    prev_time = curr_time # Updates prev_time for the next loop

    # Put the FPS onto the frame

    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), # Puts the FPS to 1 dp  at x = 10 and y = 30
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # Uses the font style Hershey Simplex, at a size of 1, Green and at a thickness of 2

    # Show result
    cv2.imshow('Object Tracker', annotated_frame)
    
    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()