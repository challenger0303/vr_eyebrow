import socket
import time

def test_stream():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(('127.0.0.1', 5555))
            request = "GET /eye/left HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: keep-alive\r\n\r\n"
            s.sendall(request.encode())
            
            bytes_data = b''
            start = time.time()
            while time.time() - start < 1.0: # read for 1 sec
                try:
                    chunk = s.recv(4096)
                    if not chunk: break
                    bytes_data += chunk
                except socket.timeout:
                    break
            
            print(f"Read {len(bytes_data)} bytes.")
            print("First 1000 bytes:")
            print(bytes_data[:1000])
            
            # Look for boundary
            parts = bytes_data.split(b'\r\n\r\n')
            print(f"Found {len(parts)} parts separated by \\r\\n\\r\\n")
            if len(parts) > 1:
                print("Part 1 (header?):", parts[0][:500])
                print("Part 2 (header?):", parts[1][:500])
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    test_stream()
