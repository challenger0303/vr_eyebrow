import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import EyebrowDataset
from model import TinyBrowNet

def test_overfit_single_batch():
    data_dir = "./data/images/"
    csv_train = "./data/train.csv"
    
    train_dataset = EyebrowDataset(csv_file=csv_train, img_dir=data_dir, is_train=False) # Turn off augmentation for this specific test
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = TinyBrowNet().to('cuda')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # 1e-4 is very safe
    
    # Grab one batch
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
