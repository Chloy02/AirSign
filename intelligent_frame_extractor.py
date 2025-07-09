import os
import json
import cv2
import mediapipe as mp
import shutil

# --- Configuration ---
JSON_FILE_PATH = 'WASL_data/WLASL_v0.3.json'
VIDEOS_ROOT_DIR = 'WASL_data/videos'
OUTPUT_DIR = 'WLASL_intelligent_frames' # New output folder
WORD_LIMIT = 10 # Let's start with 10 words

def extract_intelligent_frames():
    """
    Uses MediaPipe Hand Detection to scan through videos and save only the frames
    where at least one hand is detected.
    """
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5 # You can adjust this threshold
    )

    # Create the main output directory
    if os.path.exists(OUTPUT_DIR):
        print(f"Output directory '{OUTPUT_DIR}' already exists. Deleting it for a fresh start.")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}\n")

    # Load the JSON data
    try:
        with open(JSON_FILE_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: JSON file not found at '{JSON_FILE_PATH}'. Please check the path.")
        hands.close()
        return

    processed_word_count = 0
    
    # Loop through each word entry in the JSON
    for entry in data:
        if processed_word_count >= WORD_LIMIT:
            print(f"\nReached the limit of {WORD_LIMIT} words.")
            break

        gloss = entry['gloss']
        instances = entry['instances']
        
        class_folder_path = os.path.join(OUTPUT_DIR, gloss)
        os.makedirs(class_folder_path, exist_ok=True)
        
        print(f"--- Processing word ({processed_word_count + 1}/{WORD_LIMIT}): {gloss} ---")

        # Loop through each video for that word
        for inst in instances:
            video_id = inst['video_id']
            video_path = os.path.join(VIDEOS_ROOT_DIR, f"{video_id}.mp4")

            if not os.path.exists(video_path):
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"  - Error opening video file: {video_path}")
                continue

            saved_frame_count = 0
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break # End of video
                
                # Process the frame with MediaPipe
                # Convert the BGR image to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                # If hands are detected, save the frame
                if results.multi_hand_landmarks:
                    frame_filename = f"{gloss}_{video_id}_frame_{frame_index}.jpg"
                    frame_path = os.path.join(class_folder_path, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    saved_frame_count += 1
                
                frame_index += 1

            cap.release()
            if saved_frame_count > 0:
                print(f"  - Processed video '{video_id}': Found and saved {saved_frame_count} relevant frames.")
        
        processed_word_count += 1

    hands.close()
    print("\nProcessing complete!")

if __name__ == '__main__':
    extract_intelligent_frames()
