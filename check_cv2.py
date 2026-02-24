import cv2
import time

url = 'http://127.0.0.1:5555/eye/left?fps=60'
print('Trying cv2.VideoCapture on:', url)

cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("Cannot open cv2 capture.")
else:
    frames = 0
    start_time = time.time()
    for _ in range(300): # max 300 frames
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or failed.")
            break
        frames += 1
        
        elapsed = time.time() - start_time
        if elapsed > 5.0:
            break
            
    cap.release()
    print('\n--- RESULTS ---')
    print(f'Total Frames Received in {elapsed:.2f}s: {frames}')
    print(f'Average FPS via OpenCV: {frames/elapsed:.2f} fps')
