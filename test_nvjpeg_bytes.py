import torch
import torchvision

try:
    dummy_img = torch.randint(0, 256, (1, 64, 64), dtype=torch.uint8)
    jpg_bytes = torchvision.io.encode_jpeg(dummy_img).numpy().tobytes() # regular python bytes
    
    # Can we decode directly from bytes using frombuffer?
    tensor = torch.frombuffer(jpg_bytes, dtype=torch.uint8)
    decoded = torchvision.io.decode_jpeg(tensor, mode=torchvision.io.ImageReadMode.GRAY, device='cuda')
    print("Decode success! Shape:", decoded.shape, "Device:", decoded.device)
except Exception as e:
    print("Error:", e)
