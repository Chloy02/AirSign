import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp

# --- Configuration ---
LSTM_MODEL_PATH = 'lstm_classifier_5_words.pth'
TARGET_CLASSES = ['before', 'book', 'candy', 'chair', 'clothes']

# LSTM Hyperparameters (Must match the trained model)
INPUT_SIZE = 84
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = len(TARGET_CLASSES)

# Real-time processing config
SEQUENCE_LENGTH = 30  # Reduced for faster response
CONF_THRESHOLD = 0.3  # Lower threshold for debugging

# --- Helper function for normalization ---
def normalize_landmarks(landmarks_data):
    landmarks_data = landmarks_data.reshape((2, 21, 2))
    processed_hands = []
    for hand_landmarks in landmarks_data:
        if np.any(hand_landmarks):
            wrist = hand_landmarks[0]
            relative_landmarks = hand_landmarks - wrist
            max_dist = np.max(np.linalg.norm(relative_landmarks, axis=1))
            if max_dist > 0:
                normalized_landmarks = relative_landmarks / max_dist
            else:
                normalized_landmarks = relative_landmarks
            processed_hands.append(normalized_landmarks.flatten())
        else:
            processed_hands.append(np.zeros(21 * 2))
    return np.concatenate(processed_hands)

# --- LSTM Classifier Model Definition ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1, use_mean_pool=False):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        self.use_mean_pool = use_mean_pool

    def forward(self, x):
        outputs, (hidden, _) = self.lstm(x)
        if self.use_mean_pool:
            out = self.fc(self.dropout(outputs.mean(dim=1)))
        else:
            out = self.fc(self.dropout(hidden[-1]))
        return out

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading LSTM model on {device}.")
    print(f"Recognizing words: {TARGET_CLASSES}")

    # Load trained LSTM model
    lstm_model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, dropout=0.1, use_mean_pool=True).to(device)
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
    lstm_model.eval()
    print("Model loaded successfully.")

    # Initialize MediaPipe and Webcam
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)
    
    sequence_data = []
    prediction_text = "..."
    frame_count = 0
    hands_detected = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Extract and normalize landmarks for the current frame
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        raw_landmarks = np.zeros(21 * 2 * 2) 
        current_hands = 0
        
        if results.multi_hand_landmarks:
            current_hands = len(results.multi_hand_landmarks)
            hands_detected += 1
            
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if i >= 2: break
                landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()
                start_index = i * (21 * 2)
                raw_landmarks[start_index : start_index + len(landmarks)] = landmarks
        
        feature_vector = normalize_landmarks(raw_landmarks)
        sequence_data.append(feature_vector)
        
        # Keep the sequence at a fixed length
        if len(sequence_data) > SEQUENCE_LENGTH:
            sequence_data.pop(0)

        # Make a prediction with the LSTM once we have enough frames
        if len(sequence_data) == SEQUENCE_LENGTH:
            with torch.no_grad():
                sequence_tensor = torch.tensor(np.array(sequence_data), dtype=torch.float32).unsqueeze(0).to(device)
                outputs = lstm_model(sequence_tensor)
                
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                # Get all class probabilities for debugging
                all_probs = probabilities[0].cpu().numpy()
                
                if confidence.item() > CONF_THRESHOLD:
                    prediction_text = TARGET_CLASSES[predicted_idx.item()]
                else:
                    prediction_text = "..."
                
                # Debug info
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"\n--- Frame {frame_count} ---")
                    print(f"Hands detected: {current_hands}")
                    print(f"Sequence length: {len(sequence_data)}")
                    print(f"Prediction: {prediction_text}")
                    print(f"Confidence: {confidence.item():.3f}")
                    print("All probabilities:")
                    for i, (word, prob) in enumerate(zip(TARGET_CLASSES, all_probs)):
                        print(f"  {word}: {prob:.3f}")
        
        # Draw debug info on frame
        cv2.putText(frame, f"Prediction: {prediction_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Hands: {current_hands}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Sequence: {len(sequence_data)}/{SEQUENCE_LENGTH}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('LSTM Debug - ASL Detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print(f"\n--- Summary ---")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with hands detected: {hands_detected}")
    print(f"Hand detection rate: {hands_detected/frame_count*100:.1f}%")

if __name__ == '__main__':
    main() 