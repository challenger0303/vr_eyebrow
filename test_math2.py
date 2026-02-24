import torch
import torchvision
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

def test_pipeline():
    # Gradient image to catch coordinate misalignments
    w, h = 200, 200
    dummy_np = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            dummy_np[y, x] = int((x + y) / (w + h) * 255)
            
    img_pil = Image.fromarray(dummy_np, mode='L')
    
    # 1. Dataset math
    crop_box = (int(w * 0.15), 0, int(w * 0.85), int(h * 0.4))
    img_cropped = img_pil.crop(crop_box)
    
    base_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    tensor_dataset = base_transform(img_cropped)
    
    # 2. GUI math
    tensor_l = torch.from_numpy(dummy_np).unsqueeze(0)
    h_l, w_l = tensor_l.shape[1], tensor_l.shape[2]
    crop_l = tensor_l[:, :int(h_l*0.4), int(w_l*0.15):int(w_l*0.85)]
    crop_l_resized = F.interpolate(crop_l.unsqueeze(0).float(), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)
    tensor_gui = crop_l_resized / 127.5 - 1.0
    
    # Compare
    diff = torch.abs(tensor_dataset - tensor_gui)
    print(f"Dataset tensor shape {tensor_dataset.shape}")
    print(f"GUI tensor shape {tensor_gui.shape}")
    print(f"Max difference: {diff.max().item():.4f}")
    print(f"Mean difference: {diff.mean().item():.4f}")

if __name__ == "__main__":
    test_pipeline()
