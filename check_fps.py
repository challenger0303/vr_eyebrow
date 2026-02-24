import requests, time
url = 'http://127.0.0.1:5555/eye/left?fps=60'
print('Connecting to BrokenEye HTTP MJPEG Stream...')
try:
    r = requests.get(url, stream=True, timeout=5)
    boundary = b'--myboundary'
    bytes_data = b''
    start_t = time.time()
    frames = 0
    print('Reading stream for 5 seconds...')
    
    for chunk in r.iter_content(chunk_size=65536):
        bytes_data += chunk
        if boundary in bytes_data:
            parts = bytes_data.split(boundary)
            frames += len(parts) - 1
            bytes_data = parts[-1]
            
        if time.time() - start_t > 5.0:
            break
            
    print('\n--- RESULTS ---')
    print(f'Total Frames Received in 5s: {frames}')
    print(f'Actual Average FPS: {frames/5.0:.2f} fps')
except Exception as e:
    print('Error:', e)
