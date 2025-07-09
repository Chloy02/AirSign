from ultralytics import YOLO

def main():
    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Train the model using your local dataset
    print("Starting model training on CPU...")
    results = model.train(
       data='ASL_gestures_yolov8/data.yaml',        # The path to your data.yaml file
       epochs=10,               # Start with a low number of epochs like 10
       imgsz=640,
       project='ASL-Project',
       name='asl_cpu_run_1'
    )
    print("Training finished.")
    print("Results saved in:", results.save_dir)

if __name__ == '__main__':
    main()
