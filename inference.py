import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import TinyBrowNet
import time

class EMARegressor:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None
        
    def update(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value

def setup_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def crop_roi(image_pil):
    w, h = image_pil.size
    crop_box = (int(w * 0.15), 0, int(w * 0.85), int(h * 0.4))
    return image_pil.crop(crop_box)

def draw_tracker_ui(display_img, prediction, label_text):
    # Scale up for viewing
    display_img = cv2.resize(display_img, (400, 200))
    
    bar_height = 150
    offset = int(-prediction * (bar_height // 2))
    
    cv2.rectangle(display_img, (350, 25), (370, 175), (0, 0, 0), -1) # Background
    cv2.line(display_img, (340, 100), (380, 100), (255, 255, 255), 2) # Center line
    
    color = (0, 255, 0) if prediction > 0 else (0, 0, 255)
    cv2.circle(display_img, (360, 100 + offset), 8, color, -1)
    
    cv2.putText(display_img, f"{label_text}: {prediction:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
    return display_img

def run_dual_inference(model_path="tinybrownet_best.pth",
                       left_url="http://127.0.0.1:5555/eye/left",
                       right_url="http://127.0.0.1:5555/eye/right"):
                       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}")
    
    model = TinyBrowNet().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded.")
    except Exception as e:
        print(f"Warning: Could not load weights. Running with random intialization.")
    model.eval()
    
    transform = setup_transform()
    ema_left = EMARegressor(alpha=0.3)
    ema_right = EMARegressor(alpha=0.3)

    cap_left = cv2.VideoCapture(left_url)
    cap_right = cv2.VideoCapture(right_url)
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Could not open one or more streams.")
        return

    print("Starting dual-eye evaluation. Press 'q' to quit.")
    
    while True:
        start_time = time.time()
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()
        
        if not ret_l or not ret_r:
            break
            
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        
        # CRITICAL: The model only understands "Left Eyes". 
        # So we horizontally flip the right eye before inference!
        gray_r_flipped = cv2.flip(gray_r, 1)
        
        # To PIL & Crop
        pil_l = crop_roi(Image.fromarray(gray_l))
        pil_r = crop_roi(Image.fromarray(gray_r_flipped))
        
        # Batch tensor processing for speed
        tensor_l = transform(pil_l).unsqueeze(0).to(device)
        tensor_r = transform(pil_r).unsqueeze(0).to(device)
        batch = torch.cat([tensor_l, tensor_r], dim=0)
        
        with torch.no_grad():
            outputs = model(batch)
            raw_l = outputs[0].item()
            raw_r = outputs[1].item()
            
        smooth_l = ema_left.update(raw_l)
        smooth_r = ema_right.update(raw_r)
        
        fps = 1.0 / (time.time() - start_time)
        
        # Build UI
        ui_l = cv2.cvtColor(np.array(pil_l), cv2.COLOR_GRAY2BGR)
        ui_r = cv2.cvtColor(np.array(pil_r), cv2.COLOR_GRAY2BGR)
        
        disp_l = draw_tracker_ui(ui_l, smooth_l, "L-Brow")
        disp_r = draw_tracker_ui(ui_r, smooth_r, "R-Brow")
        
        # Combine
        combined = cv2.hconcat([disp_l, disp_r])
        cv2.putText(combined, f"FPS: {fps:.1f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow('Dual Eyebrow Tracker', combined)
        
        # OSC Example:
        # send_osc("/avatar/parameters/LeftBrow", smooth_l)
        # send_osc("/avatar/parameters/RightBrow", smooth_r)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_dual_inference()
