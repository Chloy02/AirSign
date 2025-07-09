import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# --- Configuration ---
FEATURES_DIR = 'LSTM_features'
# OUTPUT_SIZE is determined automatically below

# Model Hyperparameters
INPUT_SIZE = 5
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.001

# --- 1. Custom Dataset for loading sequences and labels ---
class SequenceDataset(Dataset):
    def __init__(self, features_dir):
        self.features_dir = features_dir
        
        # --- UPDATED: Hardcode the classes to use ---
        # This will now ONLY look for the 'before' and 'book' folders
        self.class_folders = ['before', 'book']
        
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.class_folders)}
        print("Hardcoded classes to train:", self.class_to_idx)
        
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

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path, label = self.data_files[idx]
        sequence = np.load(file_path)
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Helper function to pad sequences
def pad_collate_fn(batch):
    (sequences, labels) = zip(*batch)
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)
    return sequences_padded, labels

# --- 2. The LSTM Classifier Model ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

def main():
    # --- 3. The Training and Validation Loop ---
    print("Loading sequence dataset...")
    dataset = SequenceDataset(features_dir=FEATURES_DIR)
    
    num_classes = len(dataset.class_folders)
    if num_classes < 2:
        print(f"Error: Found fewer than 2 class folders. Need at least 2 to train a classifier.")
        return
    print(f"Dynamically determined number of classes: {num_classes}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    if len(val_dataset) == 0:
        print("Error: Not enough data to create a validation set. Please add more video sequences.")
        return

    print(f"Dataset loaded: {len(train_dataset)} training sequences, {len(val_dataset)} validation sequences.")

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nStarting Classifier training on {device} for {NUM_EPOCHS} epochs...")

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
