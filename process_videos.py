import os
import json
import cv2
import shutil

# --- Configuration ---
# Paths have been corrected based on your latest file listing.

# 1. Path to the main WLASL JSON file
JSON_FILE_PATH = 'WASL_data/WLASL_v0.3.json' # <-- CORRECTED PATH

# 2. Path to the folder containing ALL your downloaded .mp4 videos
VIDEOS_ROOT_DIR = 'WASL_data/videos' # <-- CORRECTED PATH

# 3. Path to the folder where you want to save the output frames
OUTPUT_DIR = 'WLASL_processed_frames'

# 4. Number of words to process (starting with 10)
WORD_LIMIT = 10

def process_videos():
    """
    Processes videos based on the JSON file, extracts frames, and saves them
    into folders named after their corresponding sign language word.
    """
    if os.path.exists(OUTPUT_DIR):
        print(f"Output directory '{OUTPUT_DIR}' already exists. Deleting it for a fresh start.")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}\n")

    try:
        with open(JSON_FILE_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: JSON file not found at '{JSON_FILE_PATH}'. Please check the path.")
        return

    processed_word_count = 0
    
    for entry in data:
        if processed_word_count >= WORD_LIMIT:
            print(f"\nReached the limit of {WORD_LIMIT} words.")
            break

        gloss = entry['gloss']
        instances = entry['instances']
        
        class_folder_path = os.path.join(OUTPUT_DIR, gloss)
        os.makedirs(class_folder_path, exist_ok=True)
        
        print(f"--- Processing word ({processed_word_count + 1}/{WORD_LIMIT}): {gloss} ---")

        for inst in instances:
            video_id = inst['video_id']
            video_file_name = f"{video_id}.mp4"
            video_path = os.path.join(VIDEOS_ROOT_DIR, video_file_name)

            if not os.path.exists(video_path):
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"  - Error opening video file: {video_path}")
                continue

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_filename = f"{gloss}_{video_id}_frame_{frame_count}.jpg"
                frame_path = os.path.join(class_folder_path, frame_filename)
                
                cv2.imwrite(frame_path, frame)
                frame_count += 1
            
            cap.release()
            print(f"  - Processed video '{video_id}': Extracted {frame_count} frames.")
        
        processed_word_count += 1

    print("\nProcessing complete!")


if __name__ == '__main__':
    process_videos()
