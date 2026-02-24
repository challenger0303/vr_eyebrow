import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import EyeDataset
from eye_model import EyeTrackerNet

def train_eye_model(data_dir, train_csv, val_csv, save_path="eye_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 10 
    
    # Datasets with Random Perspective augmentation for STN
    train_dataset = EyeDataset(csv_file=train_csv, img_dir=data_dir, is_train=True)
    val_dataset = EyeDataset(csv_file=val_csv, img_dir=data_dir, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = EyeTrackerNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device) # Labels are 4D now

            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # CRITICAL: Gradient clipping is just as important here to prevent NaNs on regression
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1:02d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"--> Saved best model to {save_path}")
            torch.save(model.state_dict(), save_path)
            
if __name__ == "__main__":
    if not os.path.exists("./data/eye_train.csv"):
        print("Creating mock 4D data for verification...")
        os.makedirs("./data/images", exist_ok=True)
        import pandas as pd
        import numpy as np
        import cv2
        mock_data = []
        for i in range(100):
            img_name = f"mock_eye_{i}.jpg"
            img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            cv2.imwrite(f"./data/images/{img_name}", img)
            mock_data.append({"filename": img_name, "gaze_x": 0.5, "gaze_y": -0.2, "openness": 0.9, "dilation": 0.1})
            
        df = pd.DataFrame(mock_data)
        df.to_csv("./data/eye_train.csv", index=False)
        df.to_csv("./data/eye_val.csv", index=False)
        
    train_eye_model("./data/images", "./data/eye_train.csv", "./data/eye_val.csv")
