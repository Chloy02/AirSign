from ultralytics import YOLO

def main():
    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Define the absolute path to your 2-word data.yaml file
    data_yaml_path = '/home/chloycosta/Documents/College_code/Spec_Project/Trail/ASL_Words_Project_words.yolov8/data.yaml'

    # Train the model
    print("Starting word recognition model training...")
    results = model.train(
       data=data_yaml_path,
       epochs=75,
       imgsz=640,
       project='ASL-Words-Project',
       name='run_2_words'  # <-- Updated for clarity
    )
    print("Training finished.")
    print("Results saved in:", results.save_dir)

if __name__ == '__main__':
    main()
