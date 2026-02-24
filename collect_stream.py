import cv2
import os
import pandas as pd
from pathlib import Path
import time

def collect_dual_stream(
    left_url="http://127.0.0.1:5555/eye/left", 
    right_url="http://127.0.0.1:5555/eye/right", 
    output_dir="data"
):
    """
    Connects to both Left and Right BrokenEye MJPEG streams.
    Saves BOTH eyes into the dataset simultaneously.
    CRITICAL: The Right Eye is Horizontally Flipped before saving. 
    This means the model only ever needs to learn what a "Left Eye" looks like,
    saving 50% of the model capacity/complexity.
    """
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_dir) / "train.csv"
    
    records = []
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            records = df.to_dict('records')
            print(f"Loaded {len(records)} annotations from {csv_path}")
        except:
            pass
            
    cap_left = cv2.VideoCapture(left_url)
    cap_right = cv2.VideoCapture(right_url)
    
    if not cap_left.isOpened() or not cap_right.isOpened():
        print(f"Error: Could not open one or both streams.")
        print("Ensure BrokenEye is running.")
        return

    print("====================================")
    print("DUAL-EYE DATA COLLECTION RUNNING")
    print(f"Left: {left_url}")
    print(f"Right: {right_url}")
    print("\nControls:")
    print("  Hold '8' - Record UP (Surprise)   [Target:  1.0]")
    print("  Hold '5' - Record NEUTRAL         [Target:  0.0]")
    print("  Hold '2' - Record DOWN (Frown)    [Target: -1.0]")
    print("  Press 'q' - Quit and Save")
    print("\nTip: Look into the stream, hold a face, and blink while holding the record button!")
    print("====================================")

    frame_count = len(records)
    
    while True:
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()
        
        if not ret_l or not ret_r:
            print("Stream ended or disconnected.")
            break
            
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        
        # Display
        display_l = frame_l.copy()
        display_r = frame_r.copy()
        
        cv2.putText(display_l, "LEFT EYE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_r, "RIGHT EYE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_l, f"Frames: {len(records)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Stack horizontally for preview
        combined_display = cv2.hconcat([display_l, display_r])
        cv2.imshow("BrokenEye Dual Stream", combined_display)
        
        key = cv2.waitKey(1) & 0xFF
        
        target_val = None
        label_name = ""
        
        if key == ord('8'):
            target_val = 1.0
            label_name = "up"
        elif key == ord('5'):
            target_val = 0.0
            label_name = "neutral"
        elif key == ord('2'):
            target_val = -1.0
            label_name = "down"
        elif key == ord('q'):
            break

        if target_val is not None:
            # 1. Save Left Eye (As is)
            img_name_l = f"live_left_{label_name}_{frame_count:04d}.jpg"
            img_path_l = images_dir / img_name_l
            cv2.imwrite(str(img_path_l), gray_l)
            records.append({"filename": img_name_l, "label": target_val})
            
            # 2. Save Right Eye (FLIPPED HORIZONTALLY)
            # This makes the right eye look mathematically identical to a left eye
            # so the CNN only needs to learn one generalized representations!
            flipped_r = cv2.flip(gray_r, 1) # 1 = horizontal flip
            img_name_r = f"live_right_{label_name}_{frame_count:04d}.jpg"
            img_path_r = images_dir / img_name_r
            cv2.imwrite(str(img_path_r), flipped_r)
            records.append({"filename": img_name_r, "label": target_val})
            
            frame_count += 1
            print(f"Saved L+R pair -> {target_val}")
            
            # Sleep slightly to avoid capturing 100 identical frames instantly
            time.sleep(0.05) 

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    
    if records:
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved {len(records)} total records to {csv_path}")
    else:
        print("\nNo frames were recorded.")

if __name__ == "__main__":
    collect_dual_stream()
