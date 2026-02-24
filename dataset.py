import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class EyebrowDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
                               Format: image_filename, brow_value (-1.0 to 1.0)
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_train (bool): If true, applies heavy data augmentation for training.
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.is_train = is_train
        
        # Base transform: Normalizes the tensor
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # We handle data augmentation manually in __getitem__ because we must apply the 
        # exact same geometric affine distortions to the floating-point coordinate labels.

    def __len__(self):
        return len(self.annotations)
        
    def crop_roi(self, image):
        """
        Crops the region of interest (top 40% of the image).
        Assuming input is ~600x600, this isolates the upper eyelid/brow skin.
        """
        w, h = image.size
        # Crop box: (left, upper, right, lower)
        # We discard the bottom 60% and slightly crop the sides to remove noise
        crop_box = (int(w * 0.15), 0, int(w * 0.85), int(h * 0.4))
        return image.crop(crop_box)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        
        # Load grayscale image
        image = Image.open(img_name).convert('L')
        orig_w, orig_h = image.size
        
        # Crop to Region of Interest
        image = self.crop_roi(image)
        crop_w, crop_h = image.size
        crop_left = int(orig_w * 0.15)
        crop_top = 0

        # Read labels
        brow_value = float(self.annotations.iloc[idx, 1])

        import torchvision.transforms.functional as TF
        import torchvision.transforms as T
        import random

        # Apply transforms
        if self.is_train:
            # Color Jitter
            image = T.ColorJitter(brightness=0.4, contrast=0.4)(image)
            
            # Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                
            # Affine Transform (Translate/Scale/Rotate)
            angle = random.uniform(-5, 5)
            # Translations are constrained (-10% to 10%)
            translate_x = random.uniform(-0.1, 0.1) * orig_w
            translate_y = random.uniform(-0.1, 0.1) * orig_h
            scale = random.uniform(0.9, 1.1)
            
            image = TF.affine(image, angle=angle, translate=[translate_x, translate_y], scale=scale, shear=0)

        # Final Resize to 64x64
        image = TF.resize(image, (64, 64))
        image = self.base_transform(image)
        
        # Add slight simulated sensor noise if training
        if self.is_train:
            image = image + torch.randn_like(image) * 0.05

        labels_tensor = torch.tensor([brow_value], dtype=torch.float32)

        return image, labels_tensor

# Example usage:
if __name__ == "__main__":
    print("Dataset module loaded successfully.")

class EyeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, is_train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
                               Format: image_filename, gaze_x, gaze_y, openness, dilation
            img_dir (string): Directory with all the images.
            is_train (bool): If true, applies heavy data augmentation.
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.is_train = is_train
        
        # Base transform: Resize and convert to tensor
        self.base_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Data Augmentation specific to simulating angled cameras and varied lighting
        self.aug_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            # Simulates the skewed aspect ratio and distortion of angled cameras:
            transforms.RandomPerspective(distortion_scale=0.5, p=0.7),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            # Add some simulated sensor noise
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.annotations)
        
    def crop_roi(self, image):
        """
        Crops the region of interest for eyes.
        For eye tracking, we don't discard the bottom like eyebrows.
        """
        w, h = image.size
        crop_box = (int(w * 0.15), int(h * 0.1), int(w * 0.85), int(h * 0.9))
        return image.crop(crop_box)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        
        image = Image.open(img_name).convert('L')
        image = self.crop_roi(image)

        if self.is_train and self.aug_transform:
            image = self.aug_transform(image)
        else:
            image = self.base_transform(image)

        # Get 4 continuous labels
        gaze_x = float(self.annotations.iloc[idx, 1])
        gaze_y = float(self.annotations.iloc[idx, 2])
        openness = float(self.annotations.iloc[idx, 3])
        dilation = float(self.annotations.iloc[idx, 4])
        
        labels_tensor = torch.tensor([gaze_x, gaze_y, openness, dilation], dtype=torch.float32)

        return image, labels_tensor
