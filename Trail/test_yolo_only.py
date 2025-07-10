import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os

# Configuration
YOLO_MODEL_PATH = 'ASL-Words-Project/run_2_words3/weights/best.pt'  # Path to your trained YOLO model
TARGET_CLASSES = ['before', 'book', 'candy', 'chair', 'clothes']

def main():
    print("Loading YOLOv8 model for ASL word detection...")
    
    # Load the trained YOLO model
    if os.path.exists(YOLO_MODEL_PATH):
        model = YOLO(YOLO_MODEL_PATH)
        print(f"Model loaded from: {YOLO_MODEL_PATH}")
    else:
        print(f"Error: Model not found at {YOLO_MODEL_PATH}")
        print("Available models:")
        for file in os.listdir('.'):
            if file.endswith('.pt'):
                print(f"  - {file}")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("YOLO-only ASL Detection Started!")
    print("Press 'q' to quit")
    print(f"Detecting: {TARGET_CLASSES}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Run YOLO detection
        results = model(frame, conf=0.4)  # Confidence threshold 0.5
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    if class_id < len(TARGET_CLASSES):
                        predicted_word = TARGET_CLASSES[class_id]
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"{predicted_word}: {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        print(f"Detected: {predicted_word} (conf: {confidence:.2f})")
        
        # Display frame
        cv2.imshow('YOLO ASL Detection', frame)
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("YOLO detection stopped.")

if __name__ == "__main__":
    main() 
