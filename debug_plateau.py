import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import EyebrowDataset
from model import TinyBrowNet
import os
import pandas as pd
import numpy as np

def debug_labels():
    csv_train = "./data/train.csv"
    if not os.path.exists(csv_train):
        print("CSV doesn't exist.")
        return
        
    df = pd.read_csv(csv_train)
    print("Label distribution in train.csv:")
    print(df.iloc[:, 1].value_counts())
    
    print("\nMean label:", df.iloc[:, 1].mean())
    print("Std label:", df.iloc[:, 1].std())
    
    # Let's calculate the MSE if we always predict the mean
    mean_val = df.iloc[:, 1].mean()
    mse = np.mean((df.iloc[:, 1] - mean_val) ** 2)
    print("MSE if predicting mean:", mse)

def debug_train():
    data_dir = "./data/images/"
    csv_train = "./data/train.csv"
    
    train_dataset = EyebrowDataset(csv_file=csv_train, img_dir=data_dir, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    model = TinyBrowNet().to('cuda')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4) # what we changed it to
    
    model.train()
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        print(f"Batch {i}: Loss {loss.item():.4f}")
        
        loss.backward()
        
        # Check gradients
        grad_norm = 0
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item()
                if i == 0 and "fc2" in name:
                    print(f"  Grad norm for {name}: {p.grad.norm().item():.4f}")
        print(f"Gradient norm sum: {grad_norm:.4f}")
        
        optimizer.step()
        
        # Check some outputs
        if i == 0:
            print("Outputs sample:", outputs[:5].squeeze().tolist())
            print("Labels sample:", labels[:5].squeeze().tolist())
            
        if i >= 5:
            break

if __name__ == "__main__":
    debug_labels()
    print("-" * 50)
    debug_train()
