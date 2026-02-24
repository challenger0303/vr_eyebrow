import torch
import torch.nn as nn
import torch.nn.functional as F

class EyeTrackerNet(nn.Module):
    def __init__(self):
        super(EyeTrackerNet, self).__init__()
        
        # Spatial Transformer Network (STN) Localization Network
        # This small network looks at the 64x64 image and predicts how to 
        # affine warp/rotate/zoom it to flatten angled camera views.
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 12 * 12, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        
        # Initialize the weights/biases with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Main Feature Extractor
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Fully Connected Regression Head
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.4)
        
        # Output: [GazeX, GazeY, PupilDilation, EyelidOpenness]
        self.fc2 = nn.Linear(256, 4)

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 12 * 12)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def forward(self, x):
        # 1. Spatially transform the input (un-warp angled camera)
        x_stn = self.stn(x)
        
        # 2. Extract features
        x = self.pool1(F.relu(self.bn1(self.conv1(x_stn))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # 3. Flatten
        x = x.view(x.size(0), -1)
        
        # 4. Predict 4 continuous tracking values natively
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    model = EyeTrackerNet()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EyeTrackerNet instantiated with {params} trainable parameters.")
    
    model.eval()
    dummy_input = torch.randn(1, 1, 64, 64)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
