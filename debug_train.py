import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import EyebrowDataset
import torch.nn.functional as F
import os

class TinyBrowNetDebug(nn.Module):
    def __init__(self):
        super(TinyBrowNetDebug, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # linear output!
        x = self.fc2(x)
        return x

def debug_train():
    data_dir = "./data/images/"
    csv_train = "./data/train.csv"
    
    train_dataset = EyebrowDataset(csv_file=csv_train, img_dir=data_dir, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    model = TinyBrowNetDebug().to('cuda')
    criterion = nn.MSELoss()  # Try MSE
    optimizer = optim.Adam(model.parameters(), lr=5e-4) # Lower LR as well
    
    model.train()
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        print(f"Batch {i}: Loss {loss.item():.4f}")
        
        loss.backward()
        optimizer.step()
        
        if i >= 10:
            break

if __name__ == "__main__":
    debug_train()
