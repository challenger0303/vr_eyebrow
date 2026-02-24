import cv2
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def process_videos_to_dataset(input_dir="data/raw_videos", output_dir="data", split_ratio=0.8):
    """
    Reads videos from `input_dir` and saves frames to `data/images` 
    while generating `train.csv` and `val.csv`.
    
    Expected Folder Structure for Input Videos:
    data/raw_videos/
    ├── neutral/     (Target: 0.0)
    │   ├── video1.mp4  (e.g., eyes open)
    │   └── video2.mp4  (e.g., eyes closed)
    ├── up/          (Target: 1.0)
    │   └── video3.mp4
    └── down/        (Target: -1.0)
        └── video4.mp4
    """
    
    # Establish target mapping based on folder name
    label_mapping = {
        'neutral': 0.0,
        'up': 1.0,
        'down': -1.0
    }
    
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    data_records = []
    
    print(f"Scanning for videos in {input_dir}...")
    
    for category, target_val in label_mapping.items():
        category_dir = Path(input_dir) / category
        if not category_dir.exists():
            print(f"  Warning: Directory '{category_dir}' not found. Skipping.")
            continue
            
        video_files = list(category_dir.glob("*.mp4")) + list(category_dir.glob("*.avi"))
        if not video_files:
            print(f"  No videos found in '{category_dir}'.")
            continue
            
        for video_path in video_files:
            print(f"  Processing: {video_path.name} (Label: {target_val})")
            cap = cv2.VideoCapture(str(video_path))
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Save purely the filename (e.g., "up_video3_0001.jpg")
                img_name = f"{category}_{video_path.stem}_{frame_count:04d}.jpg"
                img_path = images_dir / img_name
                
                # Assume BrokenEye output might be color, save as grayscale to save space
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(str(img_path), gray_frame)
                
                # Append to our records
                data_records.append({"filename": img_name, "label": target_val})
                frame_count += 1
                
            cap.release()
            print(f"    -> Extracted {frame_count} frames.")

    if not data_records:
        print("Error: No frames were extracted. Check your folder structure.")
        return
        
    # Shuffle and Split into Train / Val
    df = pd.DataFrame(data_records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_size = int(len(df) * split_ratio)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    # Save CSVs
    train_csv_path = Path(output_dir) / "train.csv"
    val_csv_path = Path(output_dir) / "val.csv"
    
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    print("\nExtraction Complete!")
    print(f"Total Frames: {len(df)}")
    print(f"Train Dataset: {len(train_df)} frames saved to {train_csv_path}")
    print(f"Val Dataset: {len(val_df)} frames saved to {val_csv_path}")
    print("You can now run 'python train.py'")

if __name__ == "__main__":
    # Create the expected directories if they don't exist
    for f in ['neutral', 'up', 'down']:
        Path(f"data/raw_videos/{f}").mkdir(parents=True, exist_ok=True)
        
    process_videos_to_dataset()
