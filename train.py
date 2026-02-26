import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import EyebrowDataset
from model import TinyBrowNet
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
PATIENCE = 5  # For early stopping

import time

def train_model(data_dir, csv_train, csv_val, save_path=None):
    if save_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = f"model_{timestamp}.pth"
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Datasets and Loaders
    try:
        train_dataset = EyebrowDataset(csv_file=csv_train, img_dir=data_dir, is_train=True)
        val_dataset = EyebrowDataset(csv_file=csv_val, img_dir=data_dir, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    except FileNotFoundError:
        print(f"Error: Could not find datasets at {csv_train} or {csv_val}.")
        print("Please create dummy datasets or point to real data to run training.")
        return False
    except Exception as e:
        print(f"Error: Failed to initialize datasets: {e}")
        return False

    # Initialize Model, Loss, and Optimizer
    model = TinyBrowNet().to(device)
    # MSE provides stronger gradients for pure regression after the removal of tanh
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        # Training Loop
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Simple MSE for 1D brow regression
            loss = criterion(outputs, labels)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Val Total: {val_loss:.4f}")
        
        # Early Stopping & Model Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model with Val Loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    return True

if __name__ == "__main__":
    # Ensure dataset paths exist or prompt user
    DATA_DIR = "./data/eyebrow_images/"
    TRAIN_CSV = "./data/train.csv"
    VAL_CSV = "./data/val.csv"
    
    print("VR Eyebrow Estimation - Training Loop")
    print("Make sure you have collected data following the 6-state matrix strategy.")
    train_model(DATA_DIR, TRAIN_CSV, VAL_CSV)
