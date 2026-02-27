import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyBrowNet(nn.Module):
    def __init__(self):
        super(TinyBrowNet, self).__init__()
        # Input: 1 channel (grayscale), 64x64
        # Stage 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2) # Output: 16 x 32 x 32
        
        # Stage 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2) # Output: 32 x 16 x 16
        
        # Stage 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2) # Output: 64 x 8 x 8
        
        # Stage 4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2, 2) # Output: 64 x 4 x 4
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        # Output 3 continuous values: [brow, inner, outer]
        self.fc2 = nn.Linear(128, 3)
        
    def forward(self, x):
        # Apply convolutions, batch normalizations, activation, and pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected with dropout
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        
        # Final output layer (linear, unclamped to prevent gradient vanishing)
        x = self.fc2(x)
        
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = TinyBrowNet()
    print(f"TinyBrowNet instantiated with {count_parameters(model)} trainable parameters.")
    
    # Dummy forward pass test
    model.eval() # Prevent BatchNorm ValueError on batch_size=1
    dummy_input = torch.randn(1, 1, 64, 64)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}, Value: {output.squeeze().tolist()}")
