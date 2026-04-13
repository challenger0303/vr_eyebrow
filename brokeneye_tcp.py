"""BrokenEye native TCP client for high-framerate frame reception.

BrokenEye TCP protocol on port 5555:
  Handshake:
    - Send 0x00 → receive JSON tracking data (res_id=0)
    - Send 0x01 → receive raw image frames (res_id=1)

  Frame format (res_id=1):
    [res_id: u8][length: u32 LE][width: u32 LE][height: u32 LE][bpp: u32 LE][pixels...]
    - width=200, height=200, bpp=8 (grayscale)
    - pixels: width*height bytes, 8-bit grayscale

The HTTP MJPEG endpoint (/eye/left) is capped at 30fps.
TCP raw frames bypass that limit (~60-120fps).
"""

import socket
import struct
import threading
import time
import numpy as np


class BrokenEyeTCPClient:
    """Receives eye camera frames from BrokenEye at native framerate via TCP.

    Usage:
        client = BrokenEyeTCPClient()
        client.start()
        ...
        frame_l = client.latest_left   # numpy grayscale (H,W) uint8 or None
        frame_r = client.latest_right  # numpy grayscale (H,W) uint8 or None
        print(client.fps)
        ...
        client.stop()
    """

    # Frame header: 12 bytes = width(u32) + height(u32) + bpp(u32)
    FRAME_HEADER_SIZE = 12

    def __init__(self, host="127.0.0.1", port=5555):
        self.host = host
        self.port = port
        self.latest_left = None   # grayscale numpy (H, W) uint8
        self.latest_right = None  # grayscale numpy (H, W) uint8
        self.fps = 0.0
        self._running = False
        self._thread = None
        self._sock = None

    @property
    def is_connected(self):
        return self._running and self._sock is not None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _connect(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((self.host, self.port))
        # Send 0x01 to request raw image stream
        sock.sendall(b'\x01')
        sock.settimeout(2.0)
        return sock

    def _recv_exact(self, sock, n):
        buf = b''
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed")
            buf += chunk
        return buf

    def _run(self):
        while self._running:
            try:
                print(f"[BrokenEye TCP] Connecting to {self.host}:{self.port}...")
                self._sock = self._connect()
                print("[BrokenEye TCP] Connected. Receiving raw frames...")
                self._receive_loop()
            except (ConnectionError, socket.error, OSError) as e:
                if self._running:
                    print(f"[BrokenEye TCP] Connection lost: {e}. Reconnecting in 2s...")
                    time.sleep(2)
            except Exception as e:
                if self._running:
                    print(f"[BrokenEye TCP] Error: {e}. Reconnecting in 2s...")
                    time.sleep(2)
            finally:
                if self._sock:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None

    def _receive_loop(self):
        fps_count = 0
        fps_start = time.time()
        frame_idx = 0  # alternates: even=left, odd=right (assumption)

        while self._running:
            # Read 5-byte packet header: [res_id: u8][length: u32 LE]
            header = self._recv_exact(self._sock, 5)
            res_id = header[0]
            length = struct.unpack('<I', header[1:5])[0]

            if length > 500000:
                print(f"[BrokenEye TCP] Insane payload ({length}). Reconnecting.")
                return

            payload = self._recv_exact(self._sock, length)

            if res_id == 1 and length >= self.FRAME_HEADER_SIZE:
                frame = self._decode_raw(payload)
                if frame is not None:
                    # BrokenEye sends interleaved L/R frames
                    if frame_idx % 2 == 0:
                        self.latest_left = frame
                    else:
                        self.latest_right = frame
                    frame_idx += 1
                    fps_count += 1

            # FPS counter
            now = time.time()
            if now - fps_start >= 0.5:
                self.fps = fps_count / (now - fps_start)
                fps_count = 0
                fps_start = now

    @staticmethod
    def _decode_raw(payload):
        """Decode raw pixel payload: [w:u32][h:u32][bpp:u32][pixels...]"""
        if len(payload) < 12:
            return None
        w = struct.unpack('<I', payload[0:4])[0]
        h = struct.unpack('<I', payload[4:8])[0]
        bpp = struct.unpack('<I', payload[8:12])[0]
        pixels = payload[12:]

        expected = w * h * (bpp // 8) if bpp >= 8 else w * h
        if len(pixels) < expected or w == 0 or h == 0:
            return None

        if bpp == 8:
            return np.frombuffer(pixels[:expected], dtype=np.uint8).reshape(h, w)
        elif bpp == 24:
            return np.frombuffer(pixels[:expected], dtype=np.uint8).reshape(h, w, 3)
        else:
            return np.frombuffer(pixels[:expected], dtype=np.uint8).reshape(h, w)


if __name__ == "__main__":
    print("BrokenEye TCP Raw Frame Test")
    print("Make sure BrokenEye is running on port 5555")
    print()

    client = BrokenEyeTCPClient()
    client.start()

    try:
        for i in range(10):
            time.sleep(1)
            has_l = client.latest_left is not None
            has_r = client.latest_right is not None
            if has_l:
                h, w = client.latest_left.shape[:2]
                res = f"{w}x{h}"
            else:
                res = "?"
            print(f"  [{i+1}s] FPS: {client.fps:.1f}  Left: {has_l}  Right: {has_r}  Res: {res}")
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()

    print("\nDone.")