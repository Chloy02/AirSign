import os
import numpy as np
import torch
import random
import shutil
from pathlib import Path

# Configuration
SOURCE_DIR = 'LSTM_features_landmarks'
TARGET_DIR = 'LSTM_features_landmarks_augmented'
TARGET_CLASSES = ['before', 'book', 'candy', 'chair', 'clothes']

# Augmentation settings for duplication
DUPLICATES_PER_SEQUENCE = 5  # Number of augmented copies to create per original sequence
AUGMENTATION_STRENGTHS = {
    'light': {
        'noise_std': 0.01,
        'scale_range': (0.95, 1.05),
        'translate_range': (-0.02, 0.02),
        'mirror_prob': 0.3,
        'temporal_jitter_prob': 0.2,
        'max_jitter': 1,
        'frame_dropout_prob': 0.1
    },
    'medium': {
        'noise_std': 0.02,
        'scale_range': (0.9, 1.1),
        'translate_range': (-0.05, 0.05),
        'mirror_prob': 0.5,
        'temporal_jitter_prob': 0.3,
        'max_jitter': 2,
        'frame_dropout_prob': 0.2
    },
    'strong': {
        'noise_std': 0.03,
        'scale_range': (0.85, 1.15),
        'translate_range': (-0.08, 0.08),
        'mirror_prob': 0.7,
        'temporal_jitter_prob': 0.4,
        'max_jitter': 3,
        'frame_dropout_prob': 0.3
    }
}

def augment_sequence(sequence, noise_std=0.01, scale_range=(0.9, 1.1), translate_range=(-0.05, 0.05), 
                    mirror_prob=0.5, temporal_jitter_prob=0.0, max_jitter=1, frame_dropout_prob=0.0):
    """Apply augmentation to a sequence"""
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

def duplicate_class_data(class_name, source_path, target_path, duplicates_per_sequence, augmentation_strength):
    """Duplicate all sequences in a class folder with augmentations"""
    print(f"Processing class: {class_name}")
    
    # Create target directory
    target_class_path = os.path.join(target_path, class_name)
    os.makedirs(target_class_path, exist_ok=True)
    
    # Get all .npy files in the source class directory
    source_files = [f for f in os.listdir(source_path) if f.endswith('.npy')]
    print(f"  Found {len(source_files)} original sequences")
    
    total_created = 0
    
    for i, filename in enumerate(source_files):
        source_file_path = os.path.join(source_path, filename)
        
        # Load original sequence
        original_sequence = np.load(source_file_path)
        original_sequence = torch.tensor(original_sequence, dtype=torch.float32)
        
        # Save original sequence (copy as-is)
        original_target_path = os.path.join(target_class_path, filename)
        np.save(original_target_path, original_sequence.numpy())
        total_created += 1
        
        # Create augmented duplicates
        for dup_idx in range(duplicates_per_sequence):
            # Randomly choose augmentation strength
            strength = random.choice(['light', 'medium', 'strong'])
            params = AUGMENTATION_STRENGTHS[strength]
            
            # Apply augmentation
            augmented_sequence = augment_sequence(
                original_sequence,
                noise_std=params['noise_std'],
                scale_range=params['scale_range'],
                translate_range=params['translate_range'],
                mirror_prob=params['mirror_prob'],
                temporal_jitter_prob=params['temporal_jitter_prob'],
                max_jitter=params['max_jitter'],
                frame_dropout_prob=params['frame_dropout_prob']
            )
            
            # Create filename for duplicate
            base_name = filename.replace('.npy', '')
            duplicate_filename = f"{base_name}_aug_{dup_idx+1:02d}.npy"
            duplicate_path = os.path.join(target_class_path, duplicate_filename)
            
            # Save augmented sequence
            np.save(duplicate_path, augmented_sequence.numpy())
            total_created += 1
    
    print(f"  Created {total_created} total sequences (including originals)")
    return total_created

def main():
    print("=== Data Duplication Script ===")
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Target directory: {TARGET_DIR}")
    print(f"Duplicates per sequence: {DUPLICATES_PER_SEQUENCE}")
    print(f"Total multiplier: {DUPLICATES_PER_SEQUENCE + 1}x")
    print()
    
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found!")
        return
    
    # Create target directory
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Process each class
    total_sequences = 0
    original_sequences = 0
    
    for class_name in TARGET_CLASSES:
        source_class_path = os.path.join(SOURCE_DIR, class_name)
        target_class_path = os.path.join(TARGET_DIR, class_name)
        
        if not os.path.exists(source_class_path):
            print(f"Warning: Class directory '{source_class_path}' not found, skipping...")
            continue
        
        # Count original sequences
        original_files = [f for f in os.listdir(source_class_path) if f.endswith('.npy')]
        original_sequences += len(original_files)
        
        # Duplicate data for this class
        class_total = duplicate_class_data(
            class_name, 
            source_class_path, 
            TARGET_DIR, 
            DUPLICATES_PER_SEQUENCE,
            'medium'
        )
        total_sequences += class_total
        print()
    
    # Summary
    print("=== Summary ===")
    print(f"Original sequences: {original_sequences}")
    print(f"Total sequences after duplication: {total_sequences}")
    print(f"Multiplier achieved: {total_sequences / original_sequences:.1f}x")
    print(f"New data saved to: {TARGET_DIR}")
    print()
    print("You can now update your training script to use the augmented dataset!")
    print("Change FEATURES_DIR = 'LSTM_features_landmarks_augmented' in train_5words.py")

if __name__ == "__main__":
    main() 