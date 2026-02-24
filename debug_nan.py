import torch
import torch.nn as nn
from dataset import EyebrowDataset
from model import TinyBrowNet
from torch.utils.data import DataLoader
import numpy as np

def check_nans():
    dataset = EyebrowDataset(csv_file="data/train.csv", img_dir="data/images", is_train=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    
    model = TinyBrowNet().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Register forward hooks to check for NaNs at every layer
    def get_nan_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                        print(f"NaN spotted in output {i} of {name}")
                        raise RuntimeError(f"NaN spotted in {name}")
            else:
                if isinstance(output, torch.Tensor) and torch.isnan(output).any():
                    print(f"NaN spotted in output of {name}")
                    raise RuntimeError(f"NaN spotted in {name}")
        return hook

    for name, module in model.named_modules():
        module.register_forward_hook(get_nan_hook(name))
        
    torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(5):
        for i, (images, labels) in enumerate(loader):
            if torch.isnan(images).any():
                print(f"NaN in loaded image batch {i}!")
                return
            if torch.isinf(images).any():
                print(f"Inf in loaded image batch {i}!")
                return
                
            images, labels = images.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss):
                print(f"Loss is NaN at epoch {epoch} batch {i}!")
                return
                
            try:
                loss.backward()
            except RuntimeError as e:
                print(f"Anomaly detected during backward pass at epoch {epoch} batch {i}: {e}")
                return
                
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        print(f"Epoch {epoch} completed successfully without NaNs.")

if __name__ == "__main__":
    check_nans()
