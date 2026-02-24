import torch
import torchvision
import time

try:
    print("Torch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    
    # Create a dummy valid JPEG using torchvision
    dummy_img = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    jpg_bytes = torchvision.io.encode_jpeg(dummy_img)
    
    # Test CPU decode
    t0 = time.time()
    for _ in range(100):
        img_cpu = torchvision.io.decode_jpeg(jpg_bytes, device='cpu')
    t1 = time.time()
    print(f"CPU decode: {(t1-t0)/100:.5f} sec/frame")
    
    # Test CUDA decode
    if device.type == 'cuda':
        t0 = time.time()
        for _ in range(100):
            img_cuda = torchvision.io.decode_jpeg(jpg_bytes, device='cuda')
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"CUDA decode: {(t1-t0)/100:.5f} sec/frame")
        print("CUDA Decode Success!")
except Exception as e:
    print("Error:", e)
