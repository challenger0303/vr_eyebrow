"""Lightweight MJPEG HTTP server for camera frame sharing.

Runs in a background thread. Other apps (e.g. Baballonia) can connect to
http://localhost:{port}/mjpeg to receive a live MJPEG stream, or
http://localhost:{port}/snapshot for a single JPEG frame.

This solves the Bigscreen Beyond camera exclusivity problem:
  VR Eyebrow Tracker grabs the camera → serves MJPEG → Baballonia receives it.
"""

import cv2
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

_BOUNDARY = b"mjpegstream"


class _MjpegHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving MJPEG stream and JPEG snapshots."""

    # Suppress per-request log spam
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path in ("/mjpeg", "/left", "/right"):
            self._handle_mjpeg(self.path)
        elif self.path in ("/snapshot", "/jpeg"):
            self._handle_snapshot()
        else:
            self._handle_index()

    def _handle_mjpeg(self, path="/mjpeg"):
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY.decode()}")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()

        server = self.server
        try:
            while server.streaming:
                if path == "/left":
                    jpeg = server.current_jpeg_left
                elif path == "/right":
                    jpeg = server.current_jpeg_right
                else:
                    jpeg = server.current_jpeg
                if jpeg is None:
                    time.sleep(0.033)
                    continue

                header = (
                    b"--" + _BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                    b"\r\n"
                )
                self.wfile.write(header)
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

                time.sleep(0.033)  # ~30 fps cap
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass

    def _handle_snapshot(self):
        jpeg = self.server.current_jpeg
        if jpeg is None:
            self.send_error(503, "No frame available")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(jpeg)

    def _handle_index(self):
        html = (
            b"<html><body>"
            b"<h1>VR Eyebrow Tracker - Camera Stream</h1>"
            b"<img src='/mjpeg'>"
            b"</body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)


class MjpegServer:
    """Background MJPEG HTTP server for sharing camera frames.

    Usage:
        server = MjpegServer(port=8085)
        server.start()
        ...
        server.update_frame(bgr_numpy_array)  # call every frame
        ...
        server.stop()
    """

    def __init__(self, port=8085):
        self.port = port
        self._httpd = None
        self._thread = None
        self._started = False

    @property
    def is_running(self):
        return self._started

    @property
    def client_count(self):
        """Approximate — counts active handler threads."""
        if self._httpd is None:
            return 0
        return max(0, threading.active_count() - 2)  # rough estimate

    def start(self):
        if self._started:
            return
        self._httpd = _ThreadedHTTPServer(("127.0.0.1", self.port), _MjpegHandler)
        self._httpd.streaming = True
        self._httpd.current_jpeg = None
        self._httpd.current_jpeg_left = None
        self._httpd.current_jpeg_right = None
        self._httpd.timeout = 0.5
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self._started = True
        print(f"[MJPEG Server] Serving on http://localhost:{self.port}/mjpeg")

    def _serve(self):
        while self._httpd.streaming:
            self._httpd.handle_request()

    def update_frame(self, bgr_frame):
        """Encode a BGR numpy frame to JPEG (combined stream)."""
        if self._httpd is None or bgr_frame is None:
            return
        ok, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self._httpd.current_jpeg = buf.tobytes()

    def update_frame_left(self, bgr_frame):
        """Encode left eye frame to JPEG (/left endpoint)."""
        if self._httpd is None or bgr_frame is None:
            return
        ok, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self._httpd.current_jpeg_left = buf.tobytes()

    def update_frame_right(self, bgr_frame):
        """Encode right eye frame to JPEG (/right endpoint)."""
        if self._httpd is None or bgr_frame is None:
            return
        ok, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self._httpd.current_jpeg_right = buf.tobytes()

    def stop(self):
        if not self._started:
            return
        self._httpd.streaming = False
        self._started = False
        try:
            self._httpd.server_close()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)
        self._httpd = None
        self._thread = None
        print("[MJPEG Server] Stopped")
