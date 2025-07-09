import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import random

# --- Configuration ---
FEATURES_DIR = 'LSTM_features_landmarks' # <-- CORRECTED: Points to the new landmark features
TARGET_CLASSES = ['before', 'book', 'candy', 'chair', 'clothes']

# Model Hyperparameters
INPUT_SIZE = 84      # <-- CORRECTED: From 5 to 84 (21 landmarks * 2 coords * 2 hands)
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.001

# Debug/Experiment Flags
OVERFIT_TEST = False  # Set to True to run overfitting test on 2 samples
USE_MEAN_POOL = True # Set to True to use mean pooling over LSTM outputs
USE_MLP = False      # Set to True to use MLP baseline instead of LSTM


# Augmentation parameters (easy to tune)
AUGMENT_PARAMS = {
    'mirror_prob': 0.5,
    'noise_std': 0.02,  # slightly stronger noise
    'scale_range': (0.85, 1.15),
    'translate_range': (-0.08, 0.08),
    'temporal_jitter_prob': 0.3,  # probability to jitter frames
    'max_jitter': 2,  # max number of frames to shuffle
    'frame_dropout_prob': 0.2,  # probability to drop a frame
}

def augment_sequence(sequence, mirror_prob=0.5, noise_std=0.01, scale_range=(0.9, 1.1), translate_range=(-0.05, 0.05), temporal_jitter_prob=0.0, max_jitter=1, frame_dropout_prob=0.0):
    seq = sequence.clone()
    seq_len = seq.shape[0]

    # 1. Random Mirroring (flip x-coordinates)
    if random.random() < mirror_prob:
        x_indices = torch.arange(0, 84, 2)
        seq[:, x_indices] = 1.0 - seq[:, x_indices]

    # 2. Add Gaussian Noise
    seq += torch.randn_like(seq) * noise_std

    # 3. Random Scaling (around center)
    scale = random.uniform(*scale_range)
    center = seq.mean(dim=1, keepdim=True)
    seq = (seq - center) * scale + center

    # 4. Random Translation
    translation = torch.empty((1, 84)).uniform_(*translate_range)
    seq += translation

    # 5. Temporal Jitter (shuffle a few frames)
    if random.random() < temporal_jitter_prob and seq_len > 2:
        jitter_indices = list(range(seq_len))
        for _ in range(random.randint(1, max_jitter)):
            i = random.randint(0, seq_len-2)
            jitter_indices[i], jitter_indices[i+1] = jitter_indices[i+1], jitter_indices[i]
        seq = seq[jitter_indices]

    # 6. Frame Dropout (randomly drop a frame)
    if random.random() < frame_dropout_prob and seq_len > 3:
        drop_idx = random.randint(0, seq_len-1)
        seq = torch.cat([seq[:drop_idx], seq[drop_idx+1:]], dim=0)

    return seq

# --- 1. Custom Dataset for loading sequences and labels ---
class SequenceDataset(Dataset):
    def __init__(self, features_dir, target_classes, augment=False, augment_params=None, use_mlp_features=False):
        self.features_dir = features_dir
        self.class_folders = sorted(target_classes)
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.class_folders)}
        print("Training on specific classes:", self.class_to_idx)
        self.data_files = []
        for class_name in self.class_folders:
            class_path = os.path.join(features_dir, class_name)
            if not os.path.isdir(class_path):
                print(f"Warning: Class folder not found and will be skipped: {class_path}")
                continue
            label = self.class_to_idx[class_name]
            for file_name in os.listdir(class_path):
                if file_name.endswith('.npy'):
                    self.data_files.append((os.path.join(class_path, file_name), label))
        self.augment = augment
        self.augment_params = augment_params or {}
        self.use_mlp_features = use_mlp_features

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path, label = self.data_files[idx]
        sequence = np.load(file_path)
        sequence = torch.tensor(sequence, dtype=torch.float32)
        if self.augment:
            sequence = augment_sequence(
                sequence,
                mirror_prob=self.augment_params.get('mirror_prob', 0.5),
                noise_std=self.augment_params.get('noise_std', 0.01),
                scale_range=self.augment_params.get('scale_range', (0.9, 1.1)),
                translate_range=self.augment_params.get('translate_range', (-0.05, 0.05)),
                temporal_jitter_prob=self.augment_params.get('temporal_jitter_prob', 0.0),
                max_jitter=self.augment_params.get('max_jitter', 1),
                frame_dropout_prob=self.augment_params.get('frame_dropout_prob', 0.0),
            )
        if self.use_mlp_features:
            mean = sequence.mean(dim=0)
            std = sequence.std(dim=0)
            features = torch.cat([mean, std])  # shape [168]
            return features, torch.tensor(label, dtype=torch.long)
        return sequence, torch.tensor(label, dtype=torch.long)

# Helper function to pad sequences
def pad_collate_fn(batch):
    (sequences, labels) = zip(*batch)
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)
    return sequences_padded, labels

# --- 2. The LSTM Classifier Model ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1, use_mean_pool=False):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(0.3)  # Dropout after LSTM output
        self.fc = nn.Linear(hidden_size, output_size)
        self.use_mean_pool = use_mean_pool

    def forward(self, x):
        outputs, (hidden, _) = self.lstm(x)
        if self.use_mean_pool:
            out = self.fc(self.dropout(outputs.mean(dim=1)))
        else:
            out = self.fc(self.dropout(hidden[-1]))
        return out

# --- MLP Baseline Classifier ---
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.net(x)

def main():
    # --- 3. The Training and Validation Loop ---
    print("Loading sequence dataset...")
    dataset = SequenceDataset(features_dir=FEATURES_DIR, target_classes=TARGET_CLASSES, use_mlp_features=USE_MLP)
    
    num_classes = len(dataset.class_folders)
    if num_classes == 0:
        print(f"Error: No class folders found in '{FEATURES_DIR}'.")
        return
    print(f"Dynamically determined number of classes: {num_classes}")

    # Overfit test: use only 2 samples for train and 2 for val
    if OVERFIT_TEST:
        print("[DEBUG] Running overfit test: using only 2 samples for train and 2 for val.")
        train_size = 2
        val_size = 2
        indices = np.random.permutation(len(dataset))[:4]
        train_indices = indices[:2]
        val_indices = indices[2:]
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        # For overfit test, no augmentation
        train_dataset.dataset.augment = False
    else:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        # Use augmentation for training set only
        full_indices = np.random.permutation(len(dataset))
        train_indices = full_indices[:train_size]
        val_indices = full_indices[train_size:]
        train_dataset = torch.utils.data.Subset(
            SequenceDataset(FEATURES_DIR, TARGET_CLASSES, augment=True, augment_params=AUGMENT_PARAMS, use_mlp_features=USE_MLP), train_indices)
        val_dataset = torch.utils.data.Subset(
            SequenceDataset(FEATURES_DIR, TARGET_CLASSES, augment=False, use_mlp_features=USE_MLP), val_indices)
    
    if len(val_dataset) == 0:
        print("Error: Not enough data to create a validation set. Please add more video sequences.")
        return

    print(f"Dataset loaded: {len(train_dataset)} training sequences, {len(val_dataset)} validation sequences.")

    if USE_MLP:
        # No padding needed for MLP
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
        val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if USE_MLP:
        print("[INFO] Using MLP baseline model.")
        model = MLPClassifier(input_size=168, hidden_size=64, output_size=num_classes).to(device)
    else:
        print("[INFO] Using LSTM sequence model.")
        model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes, dropout=0.1, use_mean_pool=USE_MEAN_POOL).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nStarting Classifier training on {device} for {NUM_EPOCHS} epochs...")

    # --- Visualize a batch for debugging ---
    print("[DEBUG] Visualizing a batch of input sequences and labels:")
    for sequences, labels in train_loader:
        print("Batch shape:", sequences.shape)
        print("Labels:", labels)
        if USE_MLP:
            print("First feature vector stats: min=", sequences[0].min().item(), "max=", sequences[0].max().item())
            print("First feature vector sample:", sequences[0][:10].cpu().numpy())
        else:
            print("First sequence stats: min=", sequences[0].min().item(), "max=", sequences[0].max().item())
            print("First sequence sample:", sequences[0][0][:10].cpu().numpy())
        break

    for epoch in range(NUM_EPOCHS):
        model.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}, Validation Accuracy: {accuracy:.2f}%')

    print("\nTraining complete!")
    torch.save(model.state_dict(), f'lstm_classifier_{num_classes}_words.pth')
    print(f"Model saved to lstm_classifier_{num_classes}_words.pth")

if __name__ == '__main__':
    main()
