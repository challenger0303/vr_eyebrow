import torch
import torch.nn as nn
from dataset import EyebrowDataset
from torch.utils.data import DataLoader
from model import TinyBrowNet
import torch.nn.functional as F

def check_activations():
    data_dir = "./data/images/"
    csv_train = "./data/train.csv"
    train_dataset = EyebrowDataset(csv_file=csv_train, img_dir=data_dir, is_train=False)
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    images, labels = next(iter(loader))
    
    model = TinyBrowNet()
    
    x = images
    print(f"Input: mean={x.mean().item():.4f}, std={x.std().item():.4f}, <0: {(x < 0).float().mean().item():.4f}")
    
    x = model.conv1(x)
    print(f"Conv1 before ReLU: mean={x.mean().item():.4f}, std={x.std().item():.4f}, <0: {(x < 0).float().mean().item():.4f}")
    x = F.relu(x)
    print(f"Conv1 after ReLU: mean={x.mean().item():.4f}, std={x.std().item():.4f}, ==0: {(x == 0).float().mean().item():.4f}")
    x = model.pool1(x)
    
    x = model.conv2(x)
    x = F.relu(x)
    print(f"Conv2 after ReLU: ==0: {(x == 0).float().mean().item():.4f}")
    x = model.pool2(x)
    
    x = model.conv3(x)
    x = F.relu(x)
    print(f"Conv3 after ReLU: ==0: {(x == 0).float().mean().item():.4f}")
    x = model.pool3(x)
    
    x = model.conv4(x)
    x = F.relu(x)
    print(f"Conv4 after ReLU: ==0: {(x == 0).float().mean().item():.4f}")
    x = model.pool4(x)
    
    x = x.view(x.size(0), -1)
    x = model.fc1(x)
    x = F.relu(x)
    print(f"FC1 after ReLU: ==0: {(x == 0).float().mean().item():.4f}")

if __name__ == "__main__":
    check_activations()
