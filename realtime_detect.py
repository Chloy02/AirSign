from ultralytics import YOLO

def main():
    # Load your custom-trained YOLOv8 model
    # Make sure the path to your 'best.pt' file is correct
    model = YOLO('ASL-Project/asl_cpu_run_13/weights/best.pt')

    print("Starting webcam detection...")
    print("Press 'q' to quit.")

    # Run real-time inference on the webcam feed
    model.predict(
        source=0,       # Use 0 for the default webcam
        show=True,      # Display the video feed in a window
        conf=0.2525        # Only show detections with a confidence of 50% or higher
    )

if __name__ == '__main__':
    main()
