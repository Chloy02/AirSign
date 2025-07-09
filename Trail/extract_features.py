import os
import cv2
import numpy as np
import mediapipe as mp
import shutil

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

FRAMES_INPUT_DIR = os.path.join(project_dir, 'WLASL_intelligent_frames')
FEATURES_OUTPUT_DIR = 'LSTM_features_landmarks_normalized' # New folder for normalized features
TARGET_CLASSES = ['before', 'book', 'candy', 'chair', 'clothes'] 

def normalize_landmarks(landmarks_data):
    """Normalizes landmarks to be relative to the wrist and scaled."""
    # landmarks_data is a flat array of 84 numbers
    # Reshape to (num_hands, num_landmarks, num_coords)
    landmarks_data = landmarks_data.reshape((2, 21, 2))
    
    processed_hands = []
    for hand_landmarks in landmarks_data:
        # Check if the hand is actually present (not all zeros)
        if np.any(hand_landmarks):
            # Make landmarks relative to the wrist (landmark 0)
            wrist = hand_landmarks[0]
            relative_landmarks = hand_landmarks - wrist
            
            # Scale based on the maximum distance from the wrist
            max_dist = np.max(np.linalg.norm(relative_landmarks, axis=1))
            if max_dist > 0:
                normalized_landmarks = relative_landmarks / max_dist
            else:
                normalized_landmarks = relative_landmarks
            
            processed_hands.append(normalized_landmarks.flatten())
        else:
            # If no hand, append a zero vector of the same size
            processed_hands.append(np.zeros(21 * 2))
            
    return np.concatenate(processed_hands)

def extract_normalized_features():
    """
    Extracts and normalizes hand landmarks from video frames.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    if os.path.exists(FEATURES_OUTPUT_DIR):
        print(f"Output directory '{FEATURES_OUTPUT_DIR}' already exists. Deleting it.")
        shutil.rmtree(FEATURES_OUTPUT_DIR)
    os.makedirs(FEATURES_OUTPUT_DIR)
    print(f"Created output directory: {FEATURES_OUTPUT_DIR}\n")

    for class_name in TARGET_CLASSES:
        class_input_path = os.path.join(FRAMES_INPUT_DIR, class_name)
        class_output_path = os.path.join(FEATURES_OUTPUT_DIR, class_name)
        os.makedirs(class_output_path, exist_ok=True)
        
        if not os.path.isdir(class_input_path):
            print(f"Warning: Input folder for class '{class_name}' not found. Skipping.")
            continue

        print(f"--- Processing class: {class_name} ---")

        videos = {}
        for frame_filename in os.listdir(class_input_path):
            video_id = frame_filename.split('_frame_')[0]
            if video_id not in videos: videos[video_id] = []
            videos[video_id].append(frame_filename)

        for video_id, frame_files in videos.items():
            frame_files.sort(key=lambda f: int(f.split('_frame_')[1].split('.')[0]))
            sequence_features = []

            for frame_filename in frame_files:
                frame_path = os.path.join(class_input_path, frame_filename)
                frame = cv2.imread(frame_path)
                if frame is None: continue

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                raw_landmarks = np.zeros(21 * 2 * 2) 
                if results.multi_hand_landmarks:
                    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        if i >= 2: break
                        landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()
                        start_index = i * (21 * 2)
                        raw_landmarks[start_index : start_index + len(landmarks)] = landmarks
                
                # Normalize the raw landmarks for the current frame
                normalized_feature_vector = normalize_landmarks(raw_landmarks)
                sequence_features.append(normalized_feature_vector)
            
            if sequence_features:
                output_filename = f"{video_id}.npy"
                output_path = os.path.join(class_output_path, output_filename)
                np.save(output_path, np.array(sequence_features))
                print(f"  - Saved normalized features for video '{video_id}' ({len(sequence_features)} frames)")

    hands.close()
    print("\nFeature extraction complete!")

if __name__ == '__main__':
    extract_normalized_features()
