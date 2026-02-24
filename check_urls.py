import cv2
import time

urls = [
    'http://127.0.0.1:5555/eye/left?fps=120',
    'http://127.0.0.1:5555/eye/left'
]

for url in urls:
    print(f'Testing {url}...')
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Failed to open.")
        continue
        
    frames = 0
    start = time.time()
    for _ in range(100):
        ret, _ = cap.read()
        if not ret: break
        frames += 1
        elapsed = time.time() - start
        if elapsed > 2.0: break
        
    cap.release()
    print(f'Result: {frames/elapsed:.2f} FPS\n')
