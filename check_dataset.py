import cv2
import os
import glob
import numpy as np

def check_images():
    img_dir = "./data/images/"
    images = glob.glob(img_dir + "*.jpg")
    if not images:
        print("No images found in data/images")
        return
        
    print(f"Found {len(images)} images.")
    
    for i in range(min(5, len(images))):
        img = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
        print(f"Image: {os.path.basename(images[i])} - Shape: {img.shape} - dtype: {img.dtype} - min: {np.min(img)} max: {np.max(img)}")
        if np.max(img) == 0 and np.min(img) == 0:
            print("WARNING: Image is completely black!")
            
if __name__ == "__main__":
    check_images()
