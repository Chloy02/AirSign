import cv2
from ultralytics import YOLO
import os

# --- Configuration ---
# Path to your latest and most powerful YOLO model
YOLO_MODEL_PATH = 'ASL-Words-Project/run_2_words4/weights/best.pt'

# The 8 classes your new model was trained on
TARGET_CLASSES = ['before', 'book', 'candy', 'chair', 'clothes', 'drink', 'go', 'who']


def main():
    print("Loading YOLOv8 model for ASL word detection...")
    
    # Load the trained YOLO model
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Error: Model not found at {YOLO_MODEL_PATH}")
        return
        
    model = YOLO(YOLO_MODEL_PATH)
    print(f"Model loaded from: {YOLO_MODEL_PATH}")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\nYOLO-only ASL Detection Started!")
    print("Press 'q' to quit")
    print(f"Detecting: {TARGET_CLASSES}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Run YOLO detection on the frame
        results = model(frame, conf=0.5)  # Confidence threshold
        
        # Process and display results
        annotated_frame = results[0].plot()
        
        cv2.imshow('YOLO ASL Detection', annotated_frame)
        
        # Break the loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
