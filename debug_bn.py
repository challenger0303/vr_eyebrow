import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import EyebrowDataset
import torch.nn.functional as F

class BN_TinyBrowNet(nn.Module):
    def __init__(self):
        super(BN_TinyBrowNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.fc2(x)
        return x

def test_overfit_single_batch():
    data_dir = "./data/images/"
    csv_train = "./data/train.csv"
    
    train_dataset = EyebrowDataset(csv_file=csv_train, img_dir=data_dir, is_train=False) 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = BN_TinyBrowNet().to('cuda')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    images, labels = next(iter(train_loader))
    images, labels = images.to('cuda'), labels.to('cuda')
    
    model.train()
    print(f"Target Labels:\n{labels.squeeze().tolist()[:5]}...")
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")
            
    print(f"Final predictions:\n{outputs.squeeze().tolist()[:5]}...")
    
if __name__ == "__main__":
    test_overfit_single_batch()
