import socket
import struct
import cv2
import numpy as np
import time

print("Connecting to BrokenEye RAW TCP (port 5555)...")
s = socket.socket()
s.connect(('127.0.0.1', 5555))

# Request stream
s.sendall(b'\x00')

start = time.time()
frames = 0
errors = 0

print("Reading native stream for 2 seconds...")
while time.time() - start < 2.0:
    # 5 byte header
    header = b''
    while len(header) < 5:
        ch = s.recv(5 - len(header))
        if not ch: break
        header += ch
        
    if len(header) != 5: break
    res_id = header[0]
    length = struct.unpack('<I', header[1:5])[0]
    
    if length > 500000:
        print(f"Skipping corrupt length: {length}")
        continue
        
    data = b''
    while len(data) < length:
        ch = s.recv(length - len(data))
        if not ch: break
        data += ch
        
    if res_id == 1:
        # Left eye image
        # Decode via OpenCV
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            frames += 1
        else:
            errors += 1
            
s.close()
elapsed = time.time() - start
print("\n--- NATIVE TCP RESULTS ---")
print(f"Decoded Frames in {elapsed:.2f}s: {frames}")
print(f"Decode Errors: {errors}")
if elapsed > 0:
    print(f"Actual FPS: {frames/elapsed:.2f}")
