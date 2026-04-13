"""BrokenEye MJPEG FPS diagnostic tool.

Tests multiple URL variants to find the actual framerate limit.
Run while BrokenEye is active.
"""
import cv2
import time
import requests

TESTS = [
    ("Default (no param)",       "http://127.0.0.1:5555/eye/left"),
    ("?fps=60",                  "http://127.0.0.1:5555/eye/left?fps=60"),
    ("?fps=90",                  "http://127.0.0.1:5555/eye/left?fps=90"),
    ("?fps=120",                 "http://127.0.0.1:5555/eye/left?fps=120"),
    ("?fps=0 (unlimited?)",      "http://127.0.0.1:5555/eye/left?fps=0"),
    ("Raw /eye/left/ (trailing)","http://127.0.0.1:5555/eye/left/"),
]

DURATION = 3.0  # seconds per test

print("=" * 55)
print("  BrokenEye MJPEG FPS Diagnostic")
print("=" * 55)

# First: check if BrokenEye is even running
try:
    r = requests.get("http://127.0.0.1:5555/", timeout=2)
    print(f"BrokenEye responded: {r.status_code}")
except Exception:
    print("ERROR: Cannot connect to BrokenEye on port 5555.")
    print("Make sure BrokenEye is running.")
    input("Press Enter to exit...")
    exit(1)

print()

for label, url in TESTS:
    print(f"Testing: {label}")
    print(f"  URL: {url}")
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"  FAILED to open\n")
            continue

        # Measure
        frames = 0
        start = time.perf_counter()
        frame_times = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.perf_counter()
            frames += 1
            if frames > 1:
                frame_times.append(now - prev)
            prev = now
            if now - start >= DURATION:
                break

        cap.release()
        elapsed = time.perf_counter() - start
        avg_fps = frames / elapsed if elapsed > 0 else 0

        if frame_times:
            avg_interval = sum(frame_times) / len(frame_times) * 1000
            min_interval = min(frame_times) * 1000
            max_interval = max(frame_times) * 1000
        else:
            avg_interval = min_interval = max_interval = 0

        if frame is not None:
            h, w = frame.shape[:2]
            res = f"{w}x{h}"
        else:
            res = "?"

        print(f"  Resolution: {res}")
        print(f"  Frames: {frames} in {elapsed:.1f}s = {avg_fps:.1f} FPS")
        print(f"  Interval: avg={avg_interval:.1f}ms  min={min_interval:.1f}ms  max={max_interval:.1f}ms")
        print()

    except Exception as e:
        print(f"  ERROR: {e}\n")

print("Done.")
input("Press Enter to exit...")