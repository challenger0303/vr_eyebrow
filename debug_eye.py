import torch
import torch.nn as nn
from dataset import EyeDataset
from torch.utils.data import DataLoader
from eye_model import EyeTrackerNet
import pandas as pd

def check_nans(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NAN DETECTED IN: {name}")
        return True
    return False

def main():
    print("Loading CSVs...")
    try:
        df_train = pd.read_csv("./data/eye_train.csv")
        print("Training labels contain NaNs?", df_train.isna().any().any())
    except Exception as e:
        print("Could not read eye_train.csv", e)
        return

    dataset = EyeDataset(csv_file="./data/eye_train.csv", img_dir="./data/images", is_train=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EyeTrackerNet().to(device)
    
    # Hooks to detect NaNs
    def forward_hook(module, args, output):
        if isinstance(output, tuple):
            for i, out in enumerate(output):
                check_nans(out, f"{module.__class__.__name__} output {i}")
        else:
            check_nans(output, f"{module.__class__.__name__} output")

    for name, layer in model.named_modules():
        layer.register_forward_hook(forward_hook)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    
    # torch.autograd.set_detect_anomaly(True)
    
    print("Starting loop...")
    for i, (images, labels) in enumerate(loader):
        print(f"Batch {i}")
        images, labels = images.to(device), labels.to(device)
        
        if check_nans(images, "Batch Input Images"): break
        if check_nans(labels, "Batch Input Labels"): break
        
        optimizer.zero_grad()
        
        outputs = model(images)
        if check_nans(outputs, "Model Outputs"): break
        
        loss = criterion(outputs, labels)
        if check_nans(loss, "Loss Calculation"): break
        
        loss.backward()
        
        # Check gradients
        grad_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None and check_nans(param.grad, f"Gradient for {name}"):
                grad_nan = True
        
        if grad_nan:
            break
            
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if check_nans(model.fc2.weight, "Weights after step"): break

if __name__ == "__main__":
    main()
