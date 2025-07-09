import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence # <-- Import this utility

# --- Configuration ---
FEATURES_DIR = 'LSTM_features/before' 
INPUT_SIZE = 5      
ENCODING_SIZE = 64  
NUM_LAYERS = 2
NUM_EPOCHS = 150    
BATCH_SIZE = 8
LEARNING_RATE = 0.0005

# --- 1. Custom Dataset (No changes needed here) ---
class SingleWordDataset(Dataset):
    def __init__(self, features_dir):
        self.features_dir = features_dir
        self.data_files = [os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        sequence = np.load(file_path)
        return torch.tensor(sequence, dtype=torch.float32)

# --- NEW: Helper function to pad sequences in a batch ---
def pad_collate_fn(batch):
    # batch is a list of tensors from the dataset
    # pad_sequence makes them all the same length
    sequences = pad_sequence(batch, batch_first=True, padding_value=0)
    return sequences

# --- 2. The LSTM Autoencoder Model (No changes needed here) ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, encoding_size, num_layers):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, encoding_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(encoding_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        reconstructed_sequence, _ = self.decoder(decoder_input)
        return reconstructed_sequence

def main():
    # --- 3. The Training Loop ---
    print(f"Loading sequence dataset from '{FEATURES_DIR}'...")
    if not os.path.exists(FEATURES_DIR):
        print(f"ERROR: Directory not found. Please make sure '{FEATURES_DIR}' exists.")
        return
        
    dataset = SingleWordDataset(features_dir=FEATURES_DIR)
    
    # --- UPDATED: Pass the collate function to the DataLoader ---
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
    
    print(f"Dataset loaded with {len(dataset)} sequences.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMAutoencoder(INPUT_SIZE, ENCODING_SIZE, NUM_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nStarting Autoencoder training on {device} for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for sequences in data_loader:
            sequences = sequences.to(device)
            reconstructed = model(sequences)
            loss = criterion(reconstructed, sequences)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Reconstruction Loss: {avg_loss:.6f}')

    print("\nTraining complete!")
    torch.save(model.state_dict(), 'lstm_autoencoder_before.pth')
    print("Model saved to lstm_autoencoder_before.pth")

if __name__ == '__main__':
    main()
