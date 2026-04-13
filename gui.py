import sys
import os
import platform

APP_VERSION = "1.0.0"

def _load_version_override():
    candidates = []
    try:
        exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else None
        if exe_dir:
            candidates.append(exe_dir / "VERSION.txt")
        if hasattr(sys, "_MEIPASS"):
            candidates.append(Path(sys._MEIPASS) / "VERSION.txt")
    except Exception:
        pass
    try:
        candidates.append(Path(__file__).resolve().parent / "VERSION.txt")
        if exe_dir:
            candidates.append(exe_dir / "_internal" / "VERSION.txt")
    except Exception:
        pass
    for p in candidates:
        try:
            if p and p.exists():
                ver = p.read_text(encoding="utf-8").strip()
                if ver:
                    return ver
        except Exception:
            continue
    return None

_ver_override = _load_version_override()
if _ver_override:
    APP_VERSION = _ver_override

GITHUB_REPO = "challenger0303/vr_eyebrow"
GITHUB_USER_AGENT = "VREyebrowTracker"
# ONNX Runtime MUST be imported before any other native library (torch, cv2)
# to avoid DLL initialization conflicts with CUDA runtime.
import onnxruntime  # noqa: F401 — early load to claim DLLs
import time
import re
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
from PIL import Image
import requests
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QCheckBox,
                             QLineEdit, QFrame, QGroupBox, QStyleFactory, QTabWidget,
                             QProgressBar, QFileDialog, QMessageBox, QSlider, QTableWidget, QTableWidgetItem, QHeaderView,
                             QScrollArea, QGridLayout, QComboBox, QTextEdit, QPlainTextEdit, QStackedLayout, QSizePolicy, QInputDialog, QSpinBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QPainter, QPen, QBrush, QTextCursor
# Proxy imports
import socket
import json
import struct
from pythonosc.udp_client import SimpleUDPClient

from onnx_inference import BrowNetONNX, HMDShiftTracker, get_available_providers, export_pth_to_onnx
from inference import EMARegressor, PredictiveInterpolator
from mjpeg_server import MjpegServer

class EyeVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(150, 150)
        self.gaze_x = 0.0
        self.gaze_y = 0.0
        self.openness = 1.0 # 0.0 to 1.0
        self.dilation = 0.5 # 0.0 to 1.0
        
    def update_eye(self, gx, gy, openness, dilation):
        self.gaze_x = gx
        self.gaze_y = gy
        self.openness = openness
        self.dilation = dilation
        self.update() # triggers paintEvent
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        margin = 15
        screen_w = w - margin * 2
        screen_h = h - margin * 2
        center_x = w / 2
        center_y = h / 2
        
        # Draw "Screen" background
        painter.setBrush(QBrush(QColor(40, 40, 40)))
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.drawRect(margin, margin, int(screen_w), int(screen_h))
        
        # Draw crosshairs
        painter.setPen(QPen(QColor(80, 80, 80), 1, Qt.DashLine))
        painter.drawLine(int(center_x), margin, int(center_x), int(h - margin))
        painter.drawLine(margin, int(center_y), int(w - margin), int(center_y))
        
        # Map gaze to screen coordinates
        dot_cx = center_x + (self.gaze_x * screen_w / 2)
        dot_cy = center_y - (self.gaze_y * screen_h / 2) # Y inverted
        
        # Clamp to bounds
        dot_cx = max(margin, min(dot_cx, w - margin))
        dot_cy = max(margin, min(dot_cy, h - margin))
        
        # Size based on dilation, color based on openness
        is_closed = self.openness < 0.2
        dot_radius = 6 + (self.dilation * 4)
        
        if is_closed:
            color = QColor(220, 50, 50, 180) # Red if closed
        else:
            color = QColor(50, 220, 100, 220) # Green if open
            
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawEllipse(int(dot_cx - dot_radius), int(dot_cy - dot_radius), int(dot_radius*2), int(dot_radius*2))


class LineGraphWidget(QWidget):
    def __init__(self, parent=None, max_points=120):
        super().__init__(parent)
        self.setMinimumHeight(70)
        self.setMaximumHeight(90)
        self.max_points = max_points
        self.series_l = []
        self.series_r = []

    def set_data(self, series_l, series_r):
        self.series_l = list(series_l)[-self.max_points:]
        self.series_r = list(series_r)[-self.max_points:]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, QColor(20, 20, 20))

        mid_y = h // 2
        painter.setPen(QPen(QColor(90, 90, 90), 1))
        painter.drawLine(0, mid_y, w, mid_y)

        val_l = self.series_l[-1] if self.series_l else 0.0
        val_r = self.series_r[-1] if self.series_r else 0.0

        bar_w = int(w * 0.35)
        gap = int(w * 0.05)
        left_x = int((w - (bar_w * 2 + gap)) / 2)
        right_x = left_x + bar_w + gap

        def draw_bar(x, value, label):
            value = max(-1.0, min(1.0, value))
            bar_h = int(abs(value) * (h * 0.4))
            if value >= 0:
                y = mid_y - bar_h
                height = bar_h
            else:
                y = mid_y
                height = bar_h

            painter.setBrush(QBrush(QColor(180, 180, 180)))
            painter.setPen(QPen(QColor(220, 220, 220), 1))
            painter.drawRect(x, y, bar_w, height)

            painter.setPen(QPen(QColor(230, 230, 230), 1))
            painter.drawText(x, 2, bar_w, 16, Qt.AlignCenter, label)

        draw_bar(left_x, val_l, "Left")
        draw_bar(right_x, val_r, "Right")


class CurvePreviewWidget(QWidget):
    """Mini preview of the power curve response. Shows how input maps to output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 60)
        self.gamma = 1.0
        self.boost_pos = 1.0
        self.boost_neg = 1.0

    def set_params(self, gamma, boost_pos, boost_neg):
        self.gamma = gamma
        self.boost_pos = boost_pos
        self.boost_neg = boost_neg
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        m = 8  # margin

        # Background
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))

        # Grid: center lines
        cx, cy = w // 2, h // 2
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.drawLine(m, cy, w - m, cy)
        painter.drawLine(cx, m, cx, h - m)

        # Draw curve
        pw = w - 2 * m
        ph = h - 2 * m
        steps = 60
        painter.setPen(QPen(QColor(100, 200, 255), 2))

        prev = None
        for i in range(steps + 1):
            # x goes from -1 to 1
            x = -1.0 + 2.0 * i / steps
            sign = 1.0 if x >= 0 else -1.0
            boost = self.boost_pos if x >= 0 else self.boost_neg
            y = sign * min(1.0, abs(x) ** self.gamma * boost) if x != 0 else 0.0

            # Map to pixel coordinates
            px = int(m + (x + 1.0) / 2.0 * pw)
            py = int(cy - y * (ph / 2.0))
            py = max(m, min(h - m, py))

            if prev is not None:
                painter.drawLine(prev[0], prev[1], px, py)
            prev = (px, py)

        # Labels
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        painter.drawText(m, h - 2, "-1")
        painter.drawText(w - m - 10, h - 2, "+1")
        painter.drawText(cx + 2, m + 8, "+1")
        painter.drawText(cx + 2, h - m - 2, "-1")


class ParamBarGraphWidget(QWidget):
    def __init__(self, parent=None, show_labels=True, show_values=False, show_bars=True):
        super().__init__(parent)
        self.items = []
        self.show_labels = show_labels
        self.show_values = show_values
        self.show_bars = show_bars
        self.setMinimumHeight(120)

    def set_data(self, items):
        self.items = list(items)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()

        bg = self.palette().base().color()
        painter.fillRect(0, 0, w, h, bg)

        if not self.items:
            painter.setPen(QPen(self.palette().text().color(), 1))
            painter.drawText(8, 20, "No OSC parameters")
            return

        left_margin = 10
        right_margin = 10
        top_margin = 8
        bottom_margin = 26 if self.show_labels else 8
        bar_area_w = max(10, w - left_margin - right_margin)
        bar_area_h = max(10, h - top_margin - bottom_margin)
        center_y = top_margin + bar_area_h / 2
        count = len(self.items)
        col_w = max(10, int(bar_area_w / max(1, count)))

        if self.show_bars:
            grid_pen = QPen(QColor(120, 120, 120), 1)
            painter.setPen(grid_pen)
            painter.drawLine(left_margin, int(center_y), w - right_margin, int(center_y))

        text_pen = QPen(self.palette().text().color(), 1)
        for i, (name, value) in enumerate(self.items):
            x = left_margin + i * col_w
            bar_w = max(6, int(col_w * 0.5))
            x_bar = x + (col_w - bar_w) // 2

            v = max(-1.0, min(1.0, float(value)))
            half_h = bar_area_h / 2
            bar_len = abs(v) * half_h
            if v >= 0:
                y0 = center_y - bar_len
                y1 = center_y
                color = QColor(76, 175, 80)
            else:
                y0 = center_y
                y1 = center_y + bar_len
                color = QColor(244, 67, 54)
            if self.show_bars:
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(40, 40, 40), 1))
                painter.drawRoundedRect(int(x_bar), int(y0), int(bar_w), int(max(1, y1 - y0)), 3, 3)

            if self.show_labels:
                painter.setPen(text_pen)
                label = str(name)
                painter.drawText(int(x), int(h - 12), int(col_w), 16, Qt.AlignCenter, label)

            if self.show_values:
                painter.setPen(text_pen)
                val_text = f"{v:+.2f}"
                painter.drawText(int(x), int(center_y) - 8, int(col_w), 16, Qt.AlignCenter, val_text)

class LogEmitter(QObject):
    message = pyqtSignal(str)


class StreamRedirect:
    def __init__(self, emitter):
        self.emitter = emitter

    def write(self, text):
        if text:
            self.emitter.message.emit(text)

    def flush(self):
        pass


class CameraThread(QThread):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.running = True
        self.latest_frame = None  # Store numpy array directly instead of raw bytes
        self.cap = None
        self.fps = 0.0
        self._last_ts = None
        self._fps_count = 0
        self._fps_start = None

    def run(self):
        print(f"Starting native OpenCV stream parser for: {self.source}")
        # Prefer DirectShow for local cameras to avoid MSMF instability
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.source)
        else:
            # EyeTrackVR optimization: Let C++ OpenCV backend handle the entire MJPEG un-chunking process in memory
            try:
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            except Exception:
                self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.source)
        self.cap.setExceptionMode(False)  # Turn off exception mode to prevent C++ aborts on disconnect
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        
        try:
            fail_count = 0
            while self.running:
                try:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        # OpenCV provides BGR numpy array natively
                        self.latest_frame = frame
                        now = time.time()
                        if self._fps_start is None:
                            self._fps_start = now
                            self._fps_count = 0
                        self._fps_count += 1
                        if (now - self._fps_start) >= 0.5:
                            dt = now - self._fps_start
                            if dt > 0:
                                self.fps = self._fps_count / dt
                            self._fps_start = now
                            self._fps_count = 0
                        self._last_ts = now
                        fail_count = 0
                    else:
                        print(f"Frame fetch warning for {self.source}")
                        fail_count += 1
                        # Reconnect if dropped
                        if not self.running: break
                        if fail_count >= 10:
                            try:
                                self.cap.release()
                            except Exception:
                                pass
                            self.cap = None
                            # Wait for OS to fully release the camera device
                            time.sleep(2)
                            if not self.running: break
                            if isinstance(self.source, int):
                                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                                if not self.cap.isOpened():
                                    try: self.cap.release()
                                    except Exception: pass
                                    time.sleep(0.5)
                                    self.cap = cv2.VideoCapture(self.source)
                            else:
                                try:
                                    self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                                except Exception:
                                    self.cap = cv2.VideoCapture(self.source)
                                if not self.cap.isOpened():
                                    try: self.cap.release()
                                    except Exception: pass
                                    time.sleep(0.5)
                                    self.cap = cv2.VideoCapture(self.source)
                            fail_count = 0
                        time.sleep(1)
                except Exception as e:
                    print(f"OpenCV Camera Error ({self.source}):", e)
                    if not self.running: break
                    time.sleep(1)
        finally:
            if self.cap is not None:
                self.cap.release()

    def stop(self, wait_ms=0):
        self.running = False
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        if wait_ms and wait_ms > 0:
            # Avoid UI freeze if OpenCV read blocks
            if not self.wait(wait_ms):
                try:
                    self.terminate()
                except Exception:
                    pass
                self.wait(200)


class CameraScanThread(QThread):
    result = pyqtSignal(list, list)
    error = pyqtSignal(str)

    def __init__(self, max_index=4, parent=None):
        super().__init__(parent)
        self.max_index = max_index

    def run(self):
        try:
            available = []
            for i in range(self.max_index + 1):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap is not None and cap.isOpened():
                    available.append(i)
                if cap is not None:
                    cap.release()
            names = []
            try:
                cmd = "Get-PnpDevice -Class Camera | Select-Object -ExpandProperty FriendlyName"
                result = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
                names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if not names:
                    cmd = "Get-PnpDevice -Class Image | Select-Object -ExpandProperty FriendlyName"
                    result = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
                    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            except Exception:
                names = []
            self.result.emit(available, names)
        except Exception as e:
            self.error.emit(str(e))

TRAINING_ENV_DIR = Path(os.getenv("APPDATA", ".")) / "VREyebrowTracker" / "training_python"
PYTHON_EMBED_URL = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"


def _get_training_python():
    """Find or return the training Python. Returns path or None."""
    # Check bundled training env first
    bundled = TRAINING_ENV_DIR / "python.exe"
    if bundled.exists():
        return str(bundled)
    # Check venvs near exe/source
    for base in [Path(sys.executable).parent, Path(__file__).resolve().parent]:
        for venv in ['venv_gpu', 'venv_cpu']:
            p = base / venv / 'Scripts' / 'python.exe'
            if p.exists():
                return str(p)
        for parent in [base.parent, base.parent.parent]:
            for venv in ['venv_gpu', 'venv_cpu']:
                p = parent / venv / 'Scripts' / 'python.exe'
                if p.exists():
                    return str(p)
    return None


class TrainingSetupThread(QThread):
    """Downloads and sets up Python embeddable + PyTorch for training."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def run(self):
        import zipfile
        import urllib.request

        env_dir = TRAINING_ENV_DIR
        env_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Download Python embeddable
            zip_path = env_dir / "python_embed.zip"
            if not (env_dir / "python.exe").exists():
                self.progress.emit("Downloading Python 3.10 embeddable (~12MB)...")
                urllib.request.urlretrieve(PYTHON_EMBED_URL, str(zip_path))
                self.progress.emit("Extracting Python...")
                with zipfile.ZipFile(str(zip_path), 'r') as z:
                    z.extractall(str(env_dir))
                zip_path.unlink()

                # Enable pip: uncomment 'import site' in python310._pth
                pth_file = env_dir / "python310._pth"
                if pth_file.exists():
                    content = pth_file.read_text()
                    content = content.replace("#import site", "import site")
                    pth_file.write_text(content)

            py = str(env_dir / "python.exe")

            # Step 2: Install pip
            if not (env_dir / "Scripts" / "pip.exe").exists():
                self.progress.emit("Installing pip...")
                getpip = env_dir / "get-pip.py"
                urllib.request.urlretrieve(GET_PIP_URL, str(getpip))
                subprocess.run([py, str(getpip), "--no-warn-script-location"],
                               capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                getpip.unlink(missing_ok=True)

            pip = str(env_dir / "Scripts" / "pip.exe")

            # Step 3: Install PyTorch CPU + torchvision + deps
            self.progress.emit("Installing PyTorch CPU (~200MB, one-time)...")
            subprocess.run([pip, "install", "torch", "torchvision", "--index-url",
                           "https://download.pytorch.org/whl/cpu", "--no-warn-script-location"],
                          creationflags=subprocess.CREATE_NO_WINDOW)

            self.progress.emit("Installing training dependencies...")
            subprocess.run([pip, "install", "numpy<2.0", "pandas", "tqdm", "onnx", "pillow",
                           "opencv-python-headless<4.11", "--no-warn-script-location"],
                          creationflags=subprocess.CREATE_NO_WINDOW)

            # Verify
            r = subprocess.run([py, "-c", "import torch; print('ok')"],
                              capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if r.returncode == 0 and 'ok' in r.stdout:
                self.progress.emit("Training environment ready!")
                self.finished.emit(True)
            else:
                self.progress.emit("Error: PyTorch installation failed.")
                self.finished.emit(False)

        except Exception as e:
            self.progress.emit(f"Setup error: {e}")
            self.finished.emit(False)


class TrainingThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, data_dir, train_csv, val_csv, model_dir, parent=None):
        super().__init__(parent)
        self.data_dir = str(data_dir)
        self.train_csv = str(train_csv)
        self.val_csv = str(val_csv)
        self.model_dir = str(model_dir)

    def run(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_dir = Path(self.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(model_dir / f"model_{timestamp}.pth")

        if getattr(sys, 'frozen', False):
            self._run_external(save_path)
        else:
            self._run_internal(save_path)

    def _run_internal(self, save_path):
        try:
            self.progress.emit("Training...")
            import importlib, train
            importlib.reload(train)
            from train import train_model
            ok = train_model(self.data_dir, self.train_csv, self.val_csv, save_path=save_path)
            if ok:
                try:
                    from onnx_inference import export_pth_to_onnx
                    onnx_path = save_path.rsplit('.', 1)[0] + '.onnx'
                    export_pth_to_onnx(save_path, onnx_path)
                    os.remove(save_path)
                    save_path = onnx_path
                    self.progress.emit(f"Done! {onnx_path}")
                except Exception as e:
                    self.progress.emit(f"Done! {save_path} (ONNX export failed: {e})")
            else:
                self.progress.emit("Error: Training failed.")
                save_path = ""
        except Exception as e:
            self.progress.emit(f"Error: {e}")
            save_path = ""
        finally:
            self.finished.emit(save_path)

    def _run_external(self, save_path):
        try:
            py = _get_training_python()
            if py is None:
                self.progress.emit("Error: Training environment not set up. Click 'Setup Training' first.")
                self.finished.emit("")
                return

            self.progress.emit(f"Using: {py}")

            # Find train.py
            train_script = None
            search = [Path(sys.executable).parent]
            if hasattr(sys, '_MEIPASS'):
                search.append(Path(sys._MEIPASS))
            search.extend([Path(sys.executable).parent.parent,
                           Path(sys.executable).parent.parent.parent])
            for base in search:
                if (base / 'train.py').exists():
                    train_script = str(base / 'train.py')
                    break
            if train_script is None:
                self.progress.emit("Error: train.py not found.")
                self.finished.emit("")
                return

            script_dir = str(Path(train_script).parent)
            self.progress.emit("Training in progress...")

            cmd = [py, '-u', '-c',
                   f"import sys; sys.path.insert(0,{script_dir!r}); "
                   f"from train import train_model; "
                   f"ok=train_model({self.data_dir!r},{self.train_csv!r},{self.val_csv!r},save_path={save_path!r}); "
                   f"sys.exit(0 if ok else 1)"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW)
            for line in proc.stdout:
                self.progress.emit(line.rstrip())
            proc.wait()

            if proc.returncode == 0:
                self.progress.emit(f"Training Complete! {save_path}")
                onnx_path = save_path.rsplit('.', 1)[0] + '.onnx'
                try:
                    r = subprocess.run([py, '-u', '-c',
                        f"import sys; sys.path.insert(0,{script_dir!r}); "
                        f"from export_eyebrow_onnx import export_onnx; "
                        f"from pathlib import Path; "
                        f"export_onnx(Path({save_path!r}), Path({onnx_path!r}), batch_size=2, opset=17)"],
                        timeout=60, creationflags=subprocess.CREATE_NO_WINDOW,
                        capture_output=True, text=True)
                    if r.returncode == 0 and os.path.exists(onnx_path):
                        try:
                            os.remove(save_path)
                        except Exception:
                            pass
                        save_path = onnx_path
                        self.progress.emit(f"ONNX exported: {onnx_path}")
                    else:
                        self.progress.emit(f"ONNX export failed: {r.stderr or r.stdout}")
                except Exception as e:
                    self.progress.emit(f"ONNX export failed: {e}")
            else:
                self.progress.emit("Error: Training failed.")
                save_path = ""
        except Exception as e:
            self.progress.emit(f"Error: {e}")
            save_path = ""
        finally:
            self.finished.emit(save_path)

class BrokenEyeTCPProxyThread(QThread):
    osc_msg_signal = pyqtSignal(str, str, bool)
    img_msg_signal = pyqtSignal(int, bytes) # res_id, jpeg_bytes
    def __init__(self, listen_ip="127.0.0.1", listen_port=5556, target_ip="127.0.0.1", target_port=5555):
        super().__init__()
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.target_ip = target_ip
        self.target_port = target_port
        self.server = None
        self.running = False

    def run(self):
        self.running = True
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.listen_ip, self.listen_port))
        self.server.listen(1)
        self.server.settimeout(1.0)
        
        print(f"[TCP PROXY] Listening on {self.listen_ip}:{self.listen_port}, bridging to BrokenEye on {self.target_ip}:{self.target_port}")
        
        while self.running:
            try:
                # Wait for VRCFT (TobiiAdvanced module) to connect
                client_conn, addr = self.server.accept()
                print("[TCP PROXY] VRCFT Tobii Module Connected.")
            except socket.timeout:
                continue
            except Exception:
                break
                
            client_conn.settimeout(2.0)
            target_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            target_connected = False
            
            # Keep trying to connect to BrokenEye
            while self.running and not target_connected:
                try:
                    target_conn.connect((self.target_ip, self.target_port))
                    target_conn.settimeout(2.0)
                    target_connected = True
                    print("[TCP PROXY] Connected to BrokenEye Backend.")
                except Exception as e:
                    print(f"[TCP PROXY] Waiting for BrokenEye to start on {self.target_port}...")
                    time.sleep(2)
            
            if not self.running:
                client_conn.close()
                target_conn.close()
                break
                
            self._handle_connection(client_conn, target_conn)
            
    def _handle_connection(self, client, target):
        print("[TCP PROXY] Bridging Active!")
        try:
            # VRCFT Tobii module sends 0x00 to request data stream, we must forward it
            # Remove strict timeout so it doesn't crash if VRCFT idles
            client.settimeout(None) 
            target.settimeout(None)
            
            req = client.recv(1)
            if not req: 
                print("[TCP PROXY] VRCFT sent empty request. Dropping.")
                return
            target.sendall(req)
            
            frame_count = 0
            while self.running:
                # Read 5 bytes header from BrokenEye (ID + Length)
                header = b''
                while len(header) < 5:
                    chunk = target.recv(5 - len(header))
                    if not chunk: return
                    header += chunk
                
                res_id = header[0]
                # Windows C# BitConverter relies on little-endian architecture
                length = struct.unpack('<I', header[1:5])[0] 
                
                # Check for sane length to avoid memory explosion if desynced
                if length > 100000:
                    print(f"[TCP PROXY] Received insane payload length ({length})! Desynced. Dropping.")
                    return
                
                # Read JSON payload
                data = b''
                while len(data) < length:
                    chunk = target.recv(length - len(data))
                    if not chunk: return
                    data += chunk
                    
                if res_id == 3: # JSON Payload
                    json_str = data.decode('utf-8')
                    try:
                        payload = json.loads(json_str)
                        if frame_count % 15 == 0:
                            print(f"\\n--- Frame {frame_count} ---")
                            for side in ["Left", "Right"]:
                                if side in payload:
                                    print(f"  [{side}]")
                                    for k in ["Openness", "Wide", "Squeeze", "Frown"]:
                                        if k in payload[side]:
                                            print(f"    {k:<10}: {payload[side][k]:.4f}")
                        
                        # Intercept and Drop Eyebrow Values (Squeeze and Wide are mapped to Eyebrows)
                        blocked = False
                        for side in ["Left", "Right"]:
                            if side in payload:
                                if "Squeeze" in payload[side] and payload[side]["Squeeze"] != 0.0:
                                    payload[side]["Squeeze"] = 0.0
                                    blocked = True
                                if "Frown" in payload[side] and payload[side]["Frown"] != 0.0:
                                    payload[side]["Frown"] = 0.0
                                    blocked = True
                                    
                        if frame_count % 15 == 0: # Update GUI at a slower rate to prevent lag
                            self.osc_msg_signal.emit(json_str, "", False)
                                
                        frame_count += 1
                            
                        if blocked:
                            new_json_str = json.dumps(payload)
                            new_data = new_json_str.encode('utf-8')
                            new_len = len(new_data)
                            new_header = struct.pack('<BI', res_id, new_len)
                            client.sendall(new_header + new_data)
                        else:
                            client.sendall(header + data)
                    except json.JSONDecodeError:
                        # If it's malformed JSON, just pass it through untouched
                        client.sendall(header + data)
                        
                elif res_id == 1 or res_id == 2:
                    # Native 120fps JPEG Byte Arrays (Ignored for stability to use HTTP)
                    pass
                else:
                    # Unknown packet type, just forward exactly what we got
                    client.sendall(header + data)
                    
        except socket.timeout:
            print("[TCP PROXY] Connection timed out.")
        except ConnectionResetError:
            print("[TCP PROXY] Connection reset by peer.")
        except Exception as e:
            print(f"[TCP PROXY] Bridge error: {e}")
        finally:
            client.close()
            target.close()
            print("[TCP PROXY] Bridge disconnected.")

    def stop(self):
        self.running = False
        if self.server:
            try:
                self.server.close()
            except: pass
        self.wait()


class VREyebrowTrackerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VR Eyebrow Tracker")
        self.resize(800, 600)
        QApplication.setStyle(QStyleFactory.create("Fusion"))
        
        # State
        self.is_dark_mode = True
        self.is_connected = False
        self.is_connected_left = False
        self.is_connected_right = False
        self.cam_left = None
        self.cam_right = None
        
        # OSC State
        self.osc_client = None
        self.osc_proxy = None
        self.osc_enabled = False
        self.osc_ip = "127.0.0.1"
        self.osc_port = 9000
        
        self.available_devices = self._get_available_devices()
        self.device = self.available_devices[0][1]  # provider name string
        self.model = None       # BrowNetONNX instance
        self.model_has_inner_outer = True
        self.current_model_path = "None Loaded"
        # HMD shift tracking (image-level stabilization)
        self.shift_tracker_l = HMDShiftTracker()
        self.shift_tracker_r = HMDShiftTracker()
        # MJPEG server for sharing camera with Baballonia
        self.mjpeg_server = MjpegServer()
        self.mjpeg_sharing_enabled = False
        self.ema_left = PredictiveInterpolator(smooth=0.3)
        self.ema_right = PredictiveInterpolator(smooth=0.3)
        self.ema_inner_left = PredictiveInterpolator(smooth=0.3)
        self.ema_inner_right = PredictiveInterpolator(smooth=0.3)
        self.ema_outer_left = PredictiveInterpolator(smooth=0.3)
        self.ema_outer_right = PredictiveInterpolator(smooth=0.3)
        self.sym_offset_l = 0.0
        self.sym_offset_r = 0.0
        self.sym_scale_l = 1.0
        self.sym_scale_r = 1.0
        self.sym_calibrating = False
        self.sym_phase_idx = 0
        self.sym_phase_start = 0.0
        self.sym_samples_l = []
        self.sym_samples_r = []
        self.sym_phase_results = {}
        self.sym_phases = [("Neutral", 2.0), ("Max Up", 2.0), ("Max Down", 2.0)]
        self.last_raw_brow_l = None
        self.last_raw_brow_r = None
        
        self.offset_l = 0.0
        self.offset_r = 0.0
        
        self.current_fps = 0.0
        self.last_update_time = time.time()
        # Output FPS counter (OSC send rate including interpolated frames)
        self._osc_frame_count = 0
        self._osc_fps = 0.0
        self._osc_fps_time = time.time()
        
        # Data Collection State
        self.recorded_frames = []
        appdata = os.getenv("APPDATA")
        if appdata:
            self.data_dir = Path(appdata) / "VREyebrowTracker"
        else:
            self.data_dir = Path("data")
        self.settings_path = self.data_dir / "settings.json"
        self.settings = self._load_settings()
        self.eyebrow_images_dir = self.data_dir / "eyebrow_images"
        self.csv_path = self.data_dir / "train.csv"
        self.val_csv_path = self.data_dir / "val.csv"
        self.camera_devices = []
        self._camera_scan_thread = None
        self.camera_friendly_names = []
        
        # Auto-Baseline State (time-based, frame-rate independent)
        self._baseline_window_sec = 2.0  # seconds of history to analyze
        self._baseline_stamps = []       # (timestamp, brow_l, brow_r, inner_l, inner_r, outer_l, outer_r)
        self._baseline_locked = False    # hysteresis state
        self.graph_history_l = []
        self.graph_history_r = []
        self.auto_offset_brow_l = 0.0
        self.auto_offset_brow_r = 0.0
        self.auto_offset_inner_l = 0.0
        self.auto_offset_inner_r = 0.0
        self.auto_offset_outer_l = 0.0
        self.auto_offset_outer_r = 0.0
        self._err_dialog_last = {}
        self._ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
        self.osc_debug_enabled = False
        self.osc_param_order = [
            "BrowExpressionLeft",
            "BrowExpressionRight",
        ]
        self.osc_param_all = [
            "BrowExpressionLeft", "BrowExpressionRight",
            "BrowUpLeft", "BrowUpRight",
            "BrowDownLeft", "BrowDownRight",
            "BrowUp", "BrowDown",
        ]
        self.osc_param_values = {k: 0.0 for k in self.osc_param_all}
        self.osc_param_enabled = {k: True for k in self.osc_param_all}
        self.use_combined_feed = False
        self.combined_rotate = 0
        self.hmd_profile = "DIY"
        self.gh_token = os.getenv("VREYEBROW_GH_TOKEN", "").strip()
        
        if self.csv_path.exists():
            try:
                records = pd.read_csv(self.csv_path).to_dict('records')
                # Backward-compat: upgrade legacy "label" to brow/inner/outer
                for r in records:
                    if "brow" not in r and "label" in r:
                        r["brow"] = r["label"]
                        r["inner"] = r["label"]
                        r["outer"] = r["label"]
                self.recorded_frames = [r for r in records if (self.eyebrow_images_dir / r['filename']).exists()]
                
                # Auto-heal: If missing files were removed, rebuild the CSVs safely
                if len(self.recorded_frames) < len(records) and len(self.recorded_frames) > 0:
                    df = pd.DataFrame(self.recorded_frames).sample(frac=1).reset_index(drop=True)
                    t_sz = int(len(df) * 0.8)
                    df.iloc[:t_sz].to_csv(self.csv_path, index=False)
                    df.iloc[t_sz:].to_csv(self.val_csv_path, index=False)
            except: pass
            
        self.eyebrow_images_dir.mkdir(parents=True, exist_ok=True)
        
        # UI & Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.sym_timer = QTimer(self)
        self.sym_timer.timeout.connect(self._tick_symmetry_calibration)
        
        self.init_ui()
        self.apply_theme()
        self.update_dataset_status()
        self.apply_settings()
        QTimer.singleShot(200, self.scan_cameras)

    def _get_available_devices(self):
        return get_available_providers()

    def _scan_cameras(self, max_index=4):
        available = []
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                available.append(i)
            if cap is not None:
                cap.release()
        return available

    def _get_camera_friendly_names(self):
        try:
            cmd = "Get-PnpDevice -Class Camera | Select-Object -ExpandProperty FriendlyName"
            result = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
            names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if names:
                return names
            cmd = "Get-PnpDevice -Class Image | Select-Object -ExpandProperty FriendlyName"
            result = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
            names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            return names
        except Exception:
            return []

    def _load_settings(self):
        try:
            if self.settings_path.exists():
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_settings(self):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
        except Exception:
            pass

    def _update_setting(self, key, value):
        self.settings[key] = value
        self._save_settings()

    def _set_camera_combo(self, combo, value):
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return

    def _set_hmd_combo(self, label):
        if not hasattr(self, "cmb_hmd"):
            return
        for i in range(self.cmb_hmd.count()):
            if self.cmb_hmd.itemText(i) == label:
                self.cmb_hmd.setCurrentIndex(i)
                return

    def _on_hmd_changed(self, idx):
        if not hasattr(self, "cmb_hmd"):
            return
        self.hmd_profile = self.cmb_hmd.itemText(idx)
        self._update_setting("hmd_profile", self.hmd_profile)
        self._apply_hmd_ui()

    def _apply_hmd_ui(self):
        is_bigscreen = (self.hmd_profile == "Bigscreen Beyond 2e")
        is_diy = (self.hmd_profile == "DIY")
        show_babble = is_bigscreen or is_diy
        self.use_combined_feed = is_bigscreen
        self._update_setting("combined_feed", self.use_combined_feed)
        if hasattr(self, "grp_babble"):
            self.grp_babble.setVisible(show_babble)
        if hasattr(self, "babble_url_combined"):
            self.babble_url_combined.setVisible(is_bigscreen)
        if hasattr(self, "babble_url_dual"):
            self.babble_url_dual.setVisible(is_diy)
        if is_bigscreen and hasattr(self, "cmb_cam_l"):
            # reset to selection placeholder so we don't reuse a previous camera
            self.cmb_cam_l.setCurrentIndex(0)
        # Stop streams on HMD change
        if getattr(self, "is_connected_left", False):
            try:
                self.toggle_left_connection()
            except Exception:
                pass
        if getattr(self, "is_connected_right", False):
            try:
                self.toggle_right_connection()
            except Exception:
                pass
        # Adjust camera combo label
        if hasattr(self, "cmb_cam_l") and self.cmb_cam_l.count() > 0:
            self.cmb_cam_l.setItemText(0, "Select Camera" if is_bigscreen else "URL")
        if hasattr(self, "cmb_cam_r") and self.cmb_cam_r.count() > 0:
            self.cmb_cam_r.setItemText(0, "Select Camera" if is_bigscreen else "URL")
        if hasattr(self, "chk_combined"):
            self.chk_combined.blockSignals(True)
            self.chk_combined.setChecked(self.use_combined_feed)
            self.chk_combined.blockSignals(False)
        # Single stream button when combined
        if hasattr(self, "btn_connect_right"):
            self.btn_connect_right.setVisible(not is_bigscreen)
        if hasattr(self, "btn_connect_left"):
            self.btn_connect_left.setText("Start Stream" if is_bigscreen else "Start Left Stream")
        # Hide address input, keep feed selection
        if hasattr(self, "txt_cam_l"):
            self.txt_cam_l.setVisible(not is_bigscreen)
        if hasattr(self, "txt_cam_r"):
            self.txt_cam_r.setVisible(not is_bigscreen)
        if hasattr(self, "cmb_cam_r"):
            self.cmb_cam_r.setVisible(not is_bigscreen)
        if hasattr(self, "right_cam_spacer"):
            self.right_cam_spacer.setVisible(is_bigscreen)
        # Auto-select camera for Bigscreen Beyond (Bigeye)
        if is_bigscreen and hasattr(self, "cmb_cam_l") and self.camera_devices:
            # Try to auto-select Bigeye by friendly name
            if self.camera_friendly_names:
                for idx, name in enumerate(self.camera_friendly_names[:len(self.camera_devices)]):
                    if "bigeye" in name.lower():
                        self._set_camera_combo(self.cmb_cam_l, self.camera_devices[idx])
                        return
            # Fallback: if only one device, select it
            if len(self.camera_devices) == 1:
                self._set_camera_combo(self.cmb_cam_l, self.camera_devices[0])

    def _get_github_token(self):
        token = os.getenv("VREYEBROW_GH_TOKEN", "").strip()
        if token:
            self.gh_token = token
            return token
        if self.gh_token:
            return self.gh_token
        token, ok = QInputDialog.getText(
            self,
            "GitHub Token Required",
            "Enter GitHub token (repo access) for updates:",
            QLineEdit.Password,
        )
        if ok and token.strip():
            self.gh_token = token.strip()
            self._update_setting("gh_token", self.gh_token)
            return self.gh_token
        return ""

    def set_github_token(self):
        token, ok = QInputDialog.getText(
            self,
            "Set GitHub Token",
            "Enter GitHub token (repo access) for updates:",
            QLineEdit.Password,
            self.gh_token or "",
        )
        if ok:
            self.gh_token = token.strip()
            self._update_setting("gh_token", self.gh_token)

    def _parse_version(self, tag):
        if not tag:
            return None
        m = re.search(r"(\\d+)\\.(\\d+)\\.(\\d+)", str(tag))
        if not m:
            return None
        return tuple(int(x) for x in m.groups())

    def _is_newer_version(self, latest_tag):
        current = self._parse_version(APP_VERSION)
        latest = self._parse_version(latest_tag)
        if current is None or latest is None:
            return str(latest_tag) != str(APP_VERSION)
        return latest > current

    def _select_update_asset(self, assets):
        if not assets:
            return None
        preferred = ["gui.exe", "vreyebrowtracker.exe", "VREyebrowTracker.exe"]
        for name in preferred:
            for a in assets:
                if str(a.get("name", "")).lower() == name.lower():
                    return a
        for a in assets:
            if str(a.get("name", "")).lower().endswith(".exe"):
                return a
        return None

    def check_for_updates(self):
        if not getattr(sys, "frozen", False):
            QMessageBox.information(self, "Update", "Update check is available only in the built .exe.")
            return
        token = self._get_github_token()
        if not token:
            QMessageBox.warning(self, "Update", "GitHub token is required for private repo updates.")
            return
        api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
        try:
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"token {token}",
                "User-Agent": GITHUB_USER_AGENT,
            }
            resp = requests.get(api_url, headers=headers, timeout=20)
            if resp.status_code == 404:
                # No "latest" (no releases) in private repos -> fallback to list
                list_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases?per_page=1"
                resp = requests.get(list_url, headers=headers, timeout=20)
            if resp.status_code == 401 or resp.status_code == 403:
                raise RuntimeError("Unauthorized. Token may be missing 'repo' scope.")
            if resp.status_code != 200:
                raise RuntimeError(f"GitHub API error: {resp.status_code}")
            data = resp.json()
            # If list endpoint was used, take first release
            if isinstance(data, list):
                data = data[0] if data else {}
        except Exception as e:
            self._show_error_dialog("Update Error", f"Failed to check updates:\n{e}", key="update_check")
            return

        latest_tag = data.get("tag_name") or data.get("name") or ""
        assets = data.get("assets", [])
        if not latest_tag:
            QMessageBox.warning(self, "Update", "Could not determine latest version.")
            return
        if not self._is_newer_version(latest_tag):
            QMessageBox.information(self, "Update", f"You're up to date. (Current: {APP_VERSION})")
            return
        msg = f"Update available: {latest_tag}\nCurrent: {APP_VERSION}\n\nDownload and install now?"
        if QMessageBox.question(self, "Update Available", msg) != QMessageBox.Yes:
            return
        asset = self._select_update_asset(assets)
        if not asset:
            names = [a.get("name", "") for a in assets] if assets else []
            if names:
                QMessageBox.warning(self, "Update", "No suitable .exe asset found.\nAssets:\n" + "\n".join(names))
            else:
                QMessageBox.warning(self, "Update", "No release assets found in the latest release.")
            return
        try:
            self._download_and_install_update(asset, token)
        except Exception as e:
            self._show_error_dialog("Update Error", f"Failed to update:\n{e}", key="update_apply")

    def _download_and_install_update(self, asset, token):
        exe_path = Path(sys.executable)
        exe_dir = exe_path.parent
        new_path = exe_dir / f"{exe_path.stem}.update.exe"
        asset_url = asset.get("url")
        if not asset_url:
            raise RuntimeError("Missing asset download URL.")

        with requests.get(
            asset_url,
            headers={
                "Accept": "application/octet-stream",
                "Authorization": f"token {token}",
                "User-Agent": GITHUB_USER_AGENT,
            },
            stream=True,
            timeout=60,
        ) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Download failed: {r.status_code}")
            with open(new_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        bat_path = exe_dir / "update_gui.bat"
        bat = [
            "@echo off",
            "setlocal",
            "timeout /t 1 /nobreak >nul",
            ":wait",
            f"tasklist | find /i \"{exe_path.name}\" >nul",
            "if not errorlevel 1 (",
            "  timeout /t 1 /nobreak >nul",
            "  goto wait",
            ")",
            f"move /y \"{new_path}\" \"{exe_path}\" >nul",
            f"start \"\" \"{exe_path}\"",
            "del \"%~f0\"",
        ]
        with open(bat_path, "w", encoding="utf-8") as f:
            f.write("\r\n".join(bat))

        subprocess.Popen(["cmd", "/c", str(bat_path)], creationflags=subprocess.CREATE_NO_WINDOW)
        QMessageBox.information(self, "Update", "Update is ready. The app will now restart.")
        QApplication.quit()

    def _toggle_combined_feed(self, state):
        self.use_combined_feed = (state == Qt.Checked)

    def _on_babble_port_changed(self, v):
        port = int(v) if v.isdigit() else 8085
        self._update_setting("baballonia_mjpeg_port", port)
        if hasattr(self, "txt_babble_url"):
            self.txt_babble_url.setText(f"http://localhost:{port}/mjpeg")
        if hasattr(self, "txt_babble_url_l"):
            self.txt_babble_url_l.setText(f"http://localhost:{port}/left")
            self.txt_babble_url_r.setText(f"http://localhost:{port}/right")

    def _toggle_mjpeg_sharing(self, state):
        self.mjpeg_sharing_enabled = (state == Qt.Checked)
        self._update_setting("mjpeg_sharing", self.mjpeg_sharing_enabled)
        if self.mjpeg_sharing_enabled:
            # Stop any existing server before creating a new one
            if self.mjpeg_server.is_running:
                self.mjpeg_server.stop()
            port = int(self.txt_babble_port.text()) if self.txt_babble_port.text().isdigit() else 8085
            try:
                self.mjpeg_server = MjpegServer(port=port)
                self.mjpeg_server.start()
                self.lbl_mjpeg_status.setText(f"Running on port {port}")
                self.lbl_mjpeg_status.setStyleSheet("color: #4CAF50;")
            except Exception as e:
                self.lbl_mjpeg_status.setText(f"Failed: {e}")
                self.lbl_mjpeg_status.setStyleSheet("color: #d32f2f;")
                self.mjpeg_sharing_enabled = False
                self.chk_mjpeg_share.setChecked(False)
        else:
            self.mjpeg_server.stop()
            self.lbl_mjpeg_status.setText("Off")
            self.lbl_mjpeg_status.setStyleSheet("color: #888;")

    def _apply_combined_transform(self, frame_bgr):
        if frame_bgr is None:
            return None
        rot = self.cmb_combined_rotate.currentData() if hasattr(self, "cmb_combined_rotate") else 0
        if rot == 90:
            return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
        if rot == 180:
            return cv2.rotate(frame_bgr, cv2.ROTATE_180)
        if rot == 270:
            return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame_bgr

    def _split_combined_frame(self, frame_bgr):
        if frame_bgr is None:
            return None, None
        h, w = frame_bgr.shape[:2]
        if w >= h:
            mid = w // 2
            left = frame_bgr[:, :mid].copy()
            right = frame_bgr[:, mid:].copy()
        else:
            mid = h // 2
            left = frame_bgr[:mid, :].copy()
            right = frame_bgr[mid:, :].copy()
        return left, right

    def _get_camera_source(self, combo, url_text):
        data = combo.currentData()
        if data == "baballonia_mjpeg":
            port = self.settings.get("baballonia_mjpeg_port", 8085)
            return f"http://localhost:{port}/mjpeg"
        if data == "url" or data is None:
            return url_text
        return int(data)

    def scan_cameras(self):
        if self.is_connected:
            QMessageBox.information(self, "Camera Scan", "Stop streams before rescanning devices.")
            return
        if self._camera_scan_thread and self._camera_scan_thread.isRunning():
            return
        self.btn_scan_cam.setEnabled(False)
        self.btn_scan_cam.setText("Scanning...")
        self._camera_scan_thread = CameraScanThread(parent=self)
        self._camera_scan_thread.result.connect(self._apply_camera_scan)
        self._camera_scan_thread.error.connect(self._handle_camera_scan_error)
        self._camera_scan_thread.start()

    def _apply_camera_scan(self, devices, names):
        self.camera_devices = list(devices)
        self.camera_friendly_names = list(names) if names else []
        use_names = len(self.camera_friendly_names) == len(self.camera_devices)
        left_sel = self.cmb_cam_l.currentData()
        right_sel = self.cmb_cam_r.currentData()
        
        self.cmb_cam_l.blockSignals(True)
        self.cmb_cam_r.blockSignals(True)
        
        self.cmb_cam_l.clear()
        self.cmb_cam_l.addItem("URL", "url")
        for idx, i in enumerate(self.camera_devices):
            label = f"Device {i}"
            if use_names and idx < len(self.camera_friendly_names):
                label = f"{self.camera_friendly_names[idx]}"
            self.cmb_cam_l.addItem(label, i)

        self.cmb_cam_r.clear()
        self.cmb_cam_r.addItem("URL", "url")
        for idx, i in enumerate(self.camera_devices):
            label = f"Device {i}"
            if use_names and idx < len(self.camera_friendly_names):
                label = f"{self.camera_friendly_names[idx]}"
            self.cmb_cam_r.addItem(label, i)
        
        self.cmb_cam_l.blockSignals(False)
        self.cmb_cam_r.blockSignals(False)
        
        self._set_camera_combo(self.cmb_cam_l, left_sel)
        self._set_camera_combo(self.cmb_cam_r, right_sel)

        self.btn_scan_cam.setEnabled(True)
        self.btn_scan_cam.setText("Scan Cams")
        if hasattr(self, "cmb_hmd"):
            self._apply_hmd_ui()

    def _handle_camera_scan_error(self, message):
        self.btn_scan_cam.setEnabled(True)
        self.btn_scan_cam.setText("Scan Cams")
        self._show_error_dialog("Camera Scan Error", f"Failed to scan cameras:\n{message}", key="cam_scan", interval=3.0)

    def apply_settings(self):
        s = self.settings
        if "cam_left" in s:
            self.txt_cam_l.setText(s["cam_left"])
        if "cam_right" in s:
            self.txt_cam_r.setText(s["cam_right"])
        if "cam_left_source" in s:
            self._set_camera_combo(self.cmb_cam_l, s["cam_left_source"])
        if "cam_right_source" in s:
            self._set_camera_combo(self.cmb_cam_r, s["cam_right_source"])
        if "osc_ip" in s:
            self.txt_ip.setText(s["osc_ip"])
        if "osc_port" in s:
            self.txt_port.setText(str(s["osc_port"]))
        if "smooth" in s:
            self.slider_smooth.setValue(int(s["smooth"]))
        if "sync" in s:
            self.slider_sync.setValue(int(s["sync"]))
        for k in self.osc_param_all:
            dz_key = f"deadzone_{k}"
            boost_pos_key = f"boost_pos_{k}"
            boost_neg_key = f"boost_neg_{k}"
            enable_key = f"osc_enable_{k}"
            if dz_key in s and k in self.param_deadzone_sliders:
                self.param_deadzone_sliders[k].setValue(int(s[dz_key]))
            if boost_pos_key in s and k in self.param_boost_pos_sliders:
                self.param_boost_pos_sliders[k].setValue(int(s[boost_pos_key]))
            if boost_neg_key in s and k in self.param_boost_neg_sliders:
                self.param_boost_neg_sliders[k].setValue(int(s[boost_neg_key]))
            if enable_key in s:
                self.osc_param_enabled[k] = bool(s[enable_key])
        if "auto_baseline" in s:
            self.chk_auto_baseline.setChecked(bool(s["auto_baseline"]))
        if "alpha" in s:
            self.slider_alpha.setValue(int(s["alpha"]))
        if "gh_token" in s and not self.gh_token:
            self.gh_token = str(s["gh_token"])
        if "sym_offset_l" in s:
            self.sym_offset_l = float(s["sym_offset_l"])
        if "sym_offset_r" in s:
            self.sym_offset_r = float(s["sym_offset_r"])
        if "sym_scale_l" in s:
            self.sym_scale_l = float(s["sym_scale_l"])
        if "sym_scale_r" in s:
            self.sym_scale_r = float(s["sym_scale_r"])
        if "device_provider" in s:
            # Match by provider name (stable across restarts)
            target = s["device_provider"]
            for i, (lbl, prov) in enumerate(self.available_devices):
                if prov == target:
                    self.device = prov
                    self.cmb_device.setCurrentIndex(i)
                    break
        elif "device_index" in s and isinstance(s["device_index"], int):
            idx = s["device_index"]
            if 0 <= idx < len(self.available_devices):
                self.device = self.available_devices[idx][1]
                self.cmb_device.setCurrentIndex(idx)
        if "hmd_profile" in s and hasattr(self, "cmb_hmd"):
            self._set_hmd_combo(s["hmd_profile"])
        if "combined_feed" in s:
            self.use_combined_feed = bool(s["combined_feed"])
            if hasattr(self, "chk_combined"):
                self.chk_combined.setChecked(self.use_combined_feed)
        if "combined_rotate" in s and hasattr(self, "cmb_combined_rotate"):
            self._set_camera_combo(self.cmb_combined_rotate, s["combined_rotate"])
        model_loaded = False
        if "last_model_path" in s:
            path = s["last_model_path"]
            if path and os.path.exists(path):
                model_loaded = self.load_weights(path)
                if model_loaded and hasattr(self, "lbl_current_model"):
                    self.lbl_current_model.setText(os.path.basename(path))
        # Auto-load bundled model if no model loaded
        if not model_loaded:
            search = [
                Path(sys.executable).parent / "eyebrow_model.onnx",
                Path(sys.executable).parent / "_internal" / "eyebrow_model.onnx",
                Path(__file__).resolve().parent / "eyebrow_model.onnx",
            ]
            if hasattr(sys, '_MEIPASS'):
                search.insert(0, Path(sys._MEIPASS) / "eyebrow_model.onnx")
            for candidate in search:
                if candidate.exists():
                    model_loaded = self.load_weights(str(candidate))
                    if model_loaded and hasattr(self, "lbl_current_model"):
                        self.lbl_current_model.setText(f"{candidate.name} (default)")
                    break
        if "mjpeg_sharing" in s and hasattr(self, "chk_mjpeg_share"):
            self.chk_mjpeg_share.setChecked(bool(s["mjpeg_sharing"]))
        if hasattr(self, "cmb_hmd"):
            self._apply_hmd_ui()

    def on_device_changed(self, idx):
        if idx < 0 or idx >= len(self.available_devices):
            return
        label, provider = self.available_devices[idx]
        if provider == self.device:
            return
        self.device = provider
        self._update_setting("device_index", idx)
        self._update_setting("device_provider", provider)
        # Reload ONNX models with new provider
        try:
            if self.model is not None and self.current_model_path and os.path.exists(self.current_model_path):
                self.model = BrowNetONNX(self.current_model_path, provider=self.device)
        except Exception as e:
            QMessageBox.warning(self, "Device Error", f"Failed to switch to {label}:\n{e}")
            self.device = "CPUExecutionProvider"
            cpu_idx = next((i for i, (lbl, _d) in enumerate(self.available_devices) if lbl.startswith("CPU")), None)
            if cpu_idx is not None:
                self.cmb_device.setCurrentIndex(cpu_idx)
        
    def _ensure_onnx(self, path):
        """If path is a .pth file, auto-export to .onnx and return the .onnx path."""
        path = str(path)
        if path.lower().endswith(('.pth', '.pt')):
            onnx_path = path.rsplit('.', 1)[0] + '.onnx'
            if os.path.exists(onnx_path):
                return onnx_path
            print(f"Auto-converting {path} -> {onnx_path}")
            if getattr(sys, 'frozen', False):
                # Frozen build: use external Python with torch
                py = _get_training_python()
                if py is None:
                    print("Cannot convert .pth: no Python with PyTorch found.")
                    print("Run 'Setup Training Environment' first, or load an .onnx file.")
                    return None
                # Find export script
                script_dir = None
                search = [Path(sys.executable).parent]
                if hasattr(sys, '_MEIPASS'):
                    search.append(Path(sys._MEIPASS))
                for base in search:
                    if (base / 'export_eyebrow_onnx.py').exists():
                        script_dir = str(base)
                        break
                if script_dir is None:
                    print("export_eyebrow_onnx.py not found.")
                    return None
                r = subprocess.run([py, '-u', '-c',
                    f"import sys; sys.path.insert(0,{script_dir!r}); "
                    f"from export_eyebrow_onnx import export_onnx; "
                    f"from pathlib import Path; "
                    f"export_onnx(Path({path!r}), Path({onnx_path!r}), batch_size=2, opset=17)"],
                    capture_output=True, text=True, timeout=60,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                if r.returncode == 0 and os.path.exists(onnx_path):
                    return onnx_path
                print(f"Failed to convert .pth: {r.stderr or r.stdout}")
                return None
            else:
                try:
                    export_pth_to_onnx(path, onnx_path, batch_size=2, opset=17)
                except Exception as e:
                    print(f"Failed to convert .pth to ONNX: {e}")
                    return None
            return onnx_path
        return path

    def load_weights(self, path):
        try:
            onnx_path = self._ensure_onnx(path)
            if onnx_path is None:
                return False
            self.model = BrowNetONNX(onnx_path, provider=self.device)
            self.model_has_inner_outer = self.model.output_width >= 3
            self.current_model_path = onnx_path
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # TABS
        self.tabs = QTabWidget()
        self.tab_tracker = QWidget()
        self.tab_calibration = QWidget()
        self.tab_settings = QWidget()
        self.tabs.addTab(self.tab_tracker, "Tracker")
        self.tabs.addTab(self.tab_calibration, "Training")
        self.tabs.addTab(self.tab_settings, "Settings")
        main_layout.addWidget(self.tabs)

        self.setup_tracker_tab()
        self.setup_calibration_tab()
        self.setup_settings_tab()
        self._setup_console()
        
    def setup_tracker_tab(self):
        layout = QHBoxLayout(self.tab_tracker)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # ====================
        # Left Side (Cameras)
        # ====================
        eye_layout = QVBoxLayout()
        cam_row = QHBoxLayout()
        
        self.left_eye_box = QVBoxLayout()
        
        self.lbl_l_fps = QLabel("FPS: 0")
        self.lbl_l_fps.setAlignment(Qt.AlignLeft)
        self.lbl_l_fps.setProperty("class", "muted-label")
        
        self.txt_cam_l = QLineEdit("http://127.0.0.1:5555/eye/left?fps=120")
        self.txt_cam_l.setPlaceholderText("Left Camera URL (add ?fps=120 for high framerate)")
        self.txt_cam_l.setMinimumWidth(0)
        self.txt_cam_l.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.txt_cam_l.textChanged.connect(lambda v: self._update_setting("cam_left", v))
        
        self.cmb_cam_l = QComboBox()
        self.cmb_cam_l.setMinimumWidth(52)
        self.cmb_cam_l.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_cam_l.addItem("URL", "url")
        for i in self.camera_devices:
            self.cmb_cam_l.addItem(f"Cam {i}", i)
        self.cmb_cam_l.currentIndexChanged.connect(lambda idx: self._update_setting("cam_left_source", self.cmb_cam_l.itemData(idx)))
        
        cam_l_row = QHBoxLayout()
        cam_l_row.setContentsMargins(0, 0, 0, 0)
        cam_l_row.setSpacing(6)
        cam_l_row.addWidget(self.txt_cam_l)
        cam_l_row.addWidget(self.cmb_cam_l)
        self.cam_l_row = cam_l_row
        
        self.btn_scan_cam = QPushButton("Scan Cams")
        self.btn_scan_cam.setToolTip("Rescan local camera devices")
        self.btn_scan_cam.clicked.connect(self.scan_cameras)
        self.btn_scan_cam.setFixedWidth(120)
        
        
        self.left_img_label = QLabel("Left Eye Stream")
        self.left_img_label.setFixedSize(220, 220)
        self.left_img_label.setProperty("class", "cam-label")
        self.left_img_label.setAlignment(Qt.AlignCenter)

        self.btn_connect_left = QPushButton("Start Left Stream")
        self.btn_connect_left.setProperty("class", "primary-btn")
        self.btn_connect_left.clicked.connect(self.toggle_left_connection)
        
        lbl_l = QLabel("Left Eye")
        lbl_l.setAlignment(Qt.AlignCenter)
        lbl_l.setFont(QFont("Arial", 11, QFont.Bold))
        
        self.lbl_l_brow = QLabel("Brow Slider: 0.00")
        self.lbl_l_brow.setAlignment(Qt.AlignCenter)
        
        top_left_row = QHBoxLayout()
        top_left_row.setContentsMargins(0, 0, 0, 0)
        top_left_row.setSpacing(6)
        top_left_row.addWidget(self.btn_scan_cam)
        top_left_row.addStretch(1)
        top_left_row.addWidget(self.lbl_l_fps)
        top_left_widget = QWidget()
        top_left_widget.setLayout(top_left_row)
        top_left_widget.setFixedHeight(max(self.btn_scan_cam.sizeHint().height(), 32))
        self.left_eye_box.addWidget(top_left_widget)
        self.left_eye_box.addLayout(cam_l_row)
        self.left_eye_box.addWidget(self.left_img_label)
        self.left_eye_box.addWidget(self.btn_connect_left)
        self.left_eye_box.addWidget(lbl_l)
        self.left_eye_box.addWidget(self.lbl_l_brow)
        
        # Right Eye
        self.right_eye_box = QVBoxLayout()
        
        self.lbl_r_fps = QLabel("FPS: 0")
        self.lbl_r_fps.setAlignment(Qt.AlignLeft)
        self.lbl_r_fps.setProperty("class", "muted-label")
        
        self.txt_cam_r = QLineEdit("http://127.0.0.1:5555/eye/right?fps=120")
        self.txt_cam_r.setPlaceholderText("Right Camera URL (add ?fps=120 for high framerate)")
        self.txt_cam_r.setMinimumWidth(0)
        self.txt_cam_r.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.txt_cam_r.textChanged.connect(lambda v: self._update_setting("cam_right", v))
        
        self.cmb_cam_r = QComboBox()
        self.cmb_cam_r.setMinimumWidth(52)
        self.cmb_cam_r.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_cam_r.addItem("URL", "url")
        for i in self.camera_devices:
            self.cmb_cam_r.addItem(f"Cam {i}", i)
        self.cmb_cam_r.currentIndexChanged.connect(lambda idx: self._update_setting("cam_right_source", self.cmb_cam_r.itemData(idx)))
        
        cam_r_row = QHBoxLayout()
        cam_r_row.setContentsMargins(0, 0, 0, 0)
        cam_r_row.setSpacing(6)
        cam_r_row.addWidget(self.txt_cam_r)
        cam_r_row.addWidget(self.cmb_cam_r)
        self.right_cam_spacer = QWidget()
        self.right_cam_spacer.setFixedHeight(self.txt_cam_l.sizeHint().height() if hasattr(self, "txt_cam_l") else 28)
        self.right_cam_spacer.setVisible(False)
        cam_r_row.addWidget(self.right_cam_spacer)
        self.cam_r_row = cam_r_row
        
        self.right_img_label = QLabel("Right Eye Stream")
        self.right_img_label.setFixedSize(220, 220)
        self.right_img_label.setProperty("class", "cam-label")
        self.right_img_label.setAlignment(Qt.AlignCenter)

        self.btn_connect_right = QPushButton("Start Right Stream")
        self.btn_connect_right.setProperty("class", "primary-btn")
        self.btn_connect_right.clicked.connect(self.toggle_right_connection)
        
        lbl_r = QLabel("Right Eye")
        lbl_r.setAlignment(Qt.AlignCenter)
        lbl_r.setFont(QFont("Arial", 11, QFont.Bold))
        
        self.lbl_r_brow = QLabel("Brow Slider: 0.00")
        self.lbl_r_brow.setAlignment(Qt.AlignCenter)
        
        top_right_row = QHBoxLayout()
        top_right_row.setContentsMargins(0, 0, 0, 0)
        top_right_row.setSpacing(6)
        self.right_top_spacer = QWidget()
        self.right_top_spacer.setFixedSize(self.btn_scan_cam.sizeHint())
        top_right_row.addWidget(self.right_top_spacer)
        top_right_row.addStretch(1)
        top_right_row.addWidget(self.lbl_r_fps)
        top_right_widget = QWidget()
        top_right_widget.setLayout(top_right_row)
        top_right_widget.setFixedHeight(max(self.btn_scan_cam.sizeHint().height(), 32))
        self.right_eye_box.addWidget(top_right_widget)
        self.right_eye_box.addLayout(cam_r_row)
        self.right_eye_box.addWidget(self.right_img_label)
        self.right_eye_box.addWidget(self.btn_connect_right)
        self.right_eye_box.addWidget(lbl_r)
        self.right_eye_box.addWidget(self.lbl_r_brow)
        
        cam_row.addLayout(self.left_eye_box, 1)
        cam_row.addLayout(self.right_eye_box, 1)
        eye_layout.addLayout(cam_row)
        
        # Manual Override Group (debug-only container)
        grp_manual = QGroupBox("Manual Browser Override")
        grp_manual_layout = QVBoxLayout()
        
        # Manual Override
        self.chk_manual = QCheckBox("Manual Override (Testing)")
        grp_manual_layout.addWidget(self.chk_manual) # Keep this for the main checkbox
        
        slider_layout = QHBoxLayout()
        
        # Left Slider
        vbox_l = QVBoxLayout()
        vbox_l.addWidget(QLabel("Left"))
        self.slider_l = QSlider(Qt.Horizontal)
        self.slider_l.setRange(-100, 100)
        self.slider_l.setValue(0)
        self.slider_l.setTickPosition(QSlider.TicksBelow)
        self.slider_l.valueChanged.connect(self.snap_left_slider)
        vbox_l.addWidget(self.slider_l, alignment=Qt.AlignHCenter)
        slider_layout.addLayout(vbox_l)
        
        # Right Slider
        vbox_r = QVBoxLayout()
        vbox_r.addWidget(QLabel("Right"))
        self.slider_r = QSlider(Qt.Horizontal)
        self.slider_r.setRange(-100, 100)
        self.slider_r.setValue(0)
        self.slider_r.setTickPosition(QSlider.TicksBelow)
        self.slider_r.valueChanged.connect(self.snap_right_slider)
        vbox_r.addWidget(self.slider_r, alignment=Qt.AlignHCenter)
        slider_layout.addLayout(vbox_r)
        
        grp_manual_layout.addLayout(slider_layout)
        
        grp_manual.setLayout(grp_manual_layout)

        # Inference Graph
        self.grp_graph = QGroupBox("Inference Graph")
        graph_layout = QVBoxLayout()
        self.graph_widget = LineGraphWidget()
        graph_layout.addWidget(self.graph_widget)
        self.grp_graph.setLayout(graph_layout)

        # Per-Parameter Tuning (Debug)
        self.param_deadzone_sliders = {}
        self.param_boost_pos_sliders = {}
        self.param_boost_neg_sliders = {}
        self.param_group = QGroupBox("Per-Parameter Tuning")
        param_layout = QGridLayout()
        param_layout.setSpacing(4)
        param_layout.addWidget(QLabel("Param"), 0, 0)
        param_layout.addWidget(QLabel("Curve"), 0, 1)
        param_layout.addWidget(QLabel("Boost +"), 0, 2)
        param_layout.addWidget(QLabel("Boost -"), 0, 3)
        param_layout.addWidget(QLabel(""), 0, 4)
        self._param_curve_previews = {}
        row = 1
        for k in self.osc_param_order:
            short_labels = {"BrowExpressionLeft": "BrowL", "BrowExpressionRight": "BrowR"}
            label = QLabel(short_labels.get(k, k))
            preview = CurvePreviewWidget()

            dz = QSpinBox()
            dz.setRange(0, 30)
            dz.setValue(5)
            dz.setSuffix("")
            dz.setFixedWidth(55)
            dz.setFixedHeight(24)

            boost_pos = QSpinBox()
            boost_pos.setRange(50, 300)
            boost_pos.setValue(100)
            boost_pos.setSuffix("%")
            boost_pos.setFixedWidth(65)
            boost_pos.setFixedHeight(24)

            boost_neg = QSpinBox()
            boost_neg.setRange(50, 300)
            boost_neg.setValue(100)
            boost_neg.setSuffix("%")
            boost_neg.setFixedWidth(65)
            boost_neg.setFixedHeight(24)

            def _update_preview(_, key=k):
                d = self.param_deadzone_sliders[key].value() / 100.0
                gamma = 1.0 + d * 10.0
                bp = self.param_boost_pos_sliders[key].value() / 100.0
                bn = self.param_boost_neg_sliders[key].value() / 100.0
                self._param_curve_previews[key].set_params(gamma, bp, bn)

            dz.valueChanged.connect(lambda v, key=k: self._update_setting(f"deadzone_{key}", v))
            dz.valueChanged.connect(_update_preview)
            boost_pos.valueChanged.connect(lambda v, key=k: self._update_setting(f"boost_pos_{key}", v))
            boost_pos.valueChanged.connect(_update_preview)
            boost_neg.valueChanged.connect(lambda v, key=k: self._update_setting(f"boost_neg_{key}", v))
            boost_neg.valueChanged.connect(_update_preview)

            self.param_deadzone_sliders[k] = dz
            self.param_boost_pos_sliders[k] = boost_pos
            self.param_boost_neg_sliders[k] = boost_neg
            self._param_curve_previews[k] = preview

            preview.set_params(1.0 + 5 / 100.0 * 10.0, 1.0, 1.0)

            param_layout.addWidget(label, row, 0)
            param_layout.addWidget(dz, row, 1)
            param_layout.addWidget(boost_pos, row, 2)
            param_layout.addWidget(boost_neg, row, 3)
            param_layout.addWidget(preview, row, 4)
            row += 1
        self.param_group.setLayout(param_layout)

        debug_page = QWidget()
        debug_page_layout = QVBoxLayout(debug_page)
        # Inference Graph (always visible)
        eye_layout.addWidget(self.grp_graph)

        # Debug panel (toggle visibility) — horizontal layout
        self.debug_panel = QWidget()
        debug_h_layout = QHBoxLayout()
        debug_h_layout.setContentsMargins(0, 0, 0, 0)
        debug_h_layout.setSpacing(6)

        # Left side: OSC toggles + Manual override
        debug_left = QVBoxLayout()

        grp_osc_toggles = QGroupBox("OSC Parameters")
        osc_toggle_layout = QGridLayout()
        osc_toggle_layout.setSpacing(2)
        self._osc_send_keys = [
            "BrowExpressionLeft", "BrowExpressionRight",
            "BrowUpLeft", "BrowUpRight",
            "BrowDownLeft", "BrowDownRight",
            "BrowUp", "BrowDown",
        ]
        self._osc_send_labels = {
            "BrowExpressionLeft": "ExprL", "BrowExpressionRight": "ExprR",
            "BrowUpLeft": "UpL", "BrowUpRight": "UpR",
            "BrowDownLeft": "DownL", "BrowDownRight": "DownR",
            "BrowUp": "Up", "BrowDown": "Down",
        }
        col = 0
        for key in self._osc_send_keys:
            chk = QCheckBox(self._osc_send_labels.get(key, key))
            chk.setChecked(self.osc_param_enabled.get(key, True))
            chk.stateChanged.connect(lambda state, k=key: (
                self.osc_param_enabled.__setitem__(k, state == Qt.Checked),
                self._update_setting(f"osc_enable_{k}", state == Qt.Checked)
            ))
            osc_toggle_layout.addWidget(chk, col // 2, col % 2)
            col += 1
        grp_osc_toggles.setLayout(osc_toggle_layout)
        debug_left.addWidget(grp_osc_toggles)
        debug_left.addWidget(grp_manual)
        debug_left.addStretch(1)

        # Right side: Per-Parameter Tuning (curve/boost)
        debug_right = QVBoxLayout()
        debug_right.addWidget(self.param_group)
        debug_right.addStretch(1)

        debug_h_layout.addLayout(debug_left, stretch=0)
        debug_h_layout.addLayout(debug_right, stretch=1)
        self.debug_panel.setLayout(debug_h_layout)
        self.debug_panel.setVisible(False)
        eye_layout.addWidget(self.debug_panel)

        # Debug Toggle (small, bottom-left)
        eye_layout.addStretch(1)
        debug_row = QHBoxLayout()
        self.chk_osc_debug = QCheckBox("Debug")
        self.chk_osc_debug.setProperty("class", "tiny-check")
        self.chk_osc_debug.setChecked(False)
        self.chk_osc_debug.stateChanged.connect(self._toggle_osc_debug)
        debug_row.addWidget(self.chk_osc_debug)
        debug_row.addStretch(1)
        eye_layout.addLayout(debug_row)
        
        layout.addLayout(eye_layout, stretch=2)
        
        # VLine Separator
        vline = QFrame()
        vline.setFrameShape(QFrame.VLine)
        vline.setFrameShadow(QFrame.Sunken)
        layout.addWidget(vline)
        
        # ====================
        # Right Side (Tuning)
        # ====================
        settings_panel = QVBoxLayout()

        # OSC Start/Stop (quick access)
        self.btn_osc_main = QPushButton("Start OSC Sender")
        self.btn_osc_main.setProperty("class", "success-btn")
        self.btn_osc_main.clicked.connect(self.toggle_osc)
        settings_panel.addWidget(self.btn_osc_main)

        # Model Selection Group
        grp_model = QGroupBox("Model")
        grp_model_layout = QVBoxLayout()

        btn_load_model = QPushButton("Load Eyebrow Model (.onnx / .pth)")
        btn_load_model.clicked.connect(self.browse_weights)
        grp_model_layout.addWidget(btn_load_model)

        self.lbl_current_model = QLabel("Eyebrow: None Loaded")
        self.lbl_current_model.setProperty("class", "muted-label")
        grp_model_layout.addWidget(self.lbl_current_model)

        grp_model.setLayout(grp_model_layout)
        settings_panel.addWidget(grp_model)

        # Baballonia Camera Sharing (shown for Bigscreen Beyond / DIY)
        self.grp_babble = QGroupBox("Baballonia Camera Sharing")
        grp_babble_layout = QVBoxLayout()

        share_row = QHBoxLayout()
        self.chk_mjpeg_share = QCheckBox("Share camera via MJPEG")
        self.chk_mjpeg_share.setToolTip("Re-broadcast camera frames so Baballonia can receive them.")
        self.chk_mjpeg_share.stateChanged.connect(self._toggle_mjpeg_sharing)
        share_row.addWidget(self.chk_mjpeg_share)
        share_row.addWidget(QLabel("Port:"))
        self.txt_babble_port = QLineEdit(str(self.settings.get("baballonia_mjpeg_port", 8085)))
        self.txt_babble_port.setFixedWidth(60)
        self.txt_babble_port.textChanged.connect(self._on_babble_port_changed)
        share_row.addWidget(self.txt_babble_port)
        share_row.addStretch(1)
        grp_babble_layout.addLayout(share_row)

        # Copyable URL — single (Bigscreen combined) or dual (DIY left/right)
        url_style = "background: #222; color: #4FC3F7; border: 1px solid #555; padding: 4px 8px; font-size: 13px;"
        port = self.settings.get('baballonia_mjpeg_port', 8085)

        # Combined URL (Bigscreen Beyond)
        self.babble_url_combined = QWidget()
        url_row = QHBoxLayout(self.babble_url_combined)
        url_row.setContentsMargins(0, 0, 0, 0)
        url_row.addWidget(QLabel("Address:"))
        self.txt_babble_url = QLineEdit(f"http://localhost:{port}/mjpeg")
        self.txt_babble_url.setReadOnly(True)
        self.txt_babble_url.setMinimumHeight(28)
        self.txt_babble_url.setStyleSheet(url_style)
        self.txt_babble_url.mousePressEvent = lambda e: self.txt_babble_url.selectAll()
        url_row.addWidget(self.txt_babble_url, stretch=1)
        btn_copy = QPushButton("Copy")
        btn_copy.setFixedWidth(50)
        btn_copy.clicked.connect(lambda: QApplication.clipboard().setText(self.txt_babble_url.text()))
        url_row.addWidget(btn_copy)
        grp_babble_layout.addWidget(self.babble_url_combined)

        # Dual URL (DIY — left/right separate)
        self.babble_url_dual = QWidget()
        dual_layout = QVBoxLayout(self.babble_url_dual)
        dual_layout.setContentsMargins(0, 0, 0, 0)
        dual_layout.setSpacing(4)

        left_row = QHBoxLayout()
        left_row.addWidget(QLabel("Left:"))
        self.txt_babble_url_l = QLineEdit(f"http://localhost:{port}/left")
        self.txt_babble_url_l.setReadOnly(True)
        self.txt_babble_url_l.setMinimumHeight(28)
        self.txt_babble_url_l.setStyleSheet(url_style)
        self.txt_babble_url_l.mousePressEvent = lambda e: self.txt_babble_url_l.selectAll()
        left_row.addWidget(self.txt_babble_url_l, stretch=1)
        btn_copy_l = QPushButton("Copy")
        btn_copy_l.setFixedWidth(50)
        btn_copy_l.clicked.connect(lambda: QApplication.clipboard().setText(self.txt_babble_url_l.text()))
        left_row.addWidget(btn_copy_l)
        dual_layout.addLayout(left_row)

        right_row = QHBoxLayout()
        right_row.addWidget(QLabel("Right:"))
        self.txt_babble_url_r = QLineEdit(f"http://localhost:{port}/right")
        self.txt_babble_url_r.setReadOnly(True)
        self.txt_babble_url_r.setMinimumHeight(28)
        self.txt_babble_url_r.setStyleSheet(url_style)
        self.txt_babble_url_r.mousePressEvent = lambda e: self.txt_babble_url_r.selectAll()
        right_row.addWidget(self.txt_babble_url_r, stretch=1)
        btn_copy_r = QPushButton("Copy")
        btn_copy_r.setFixedWidth(50)
        btn_copy_r.clicked.connect(lambda: QApplication.clipboard().setText(self.txt_babble_url_r.text()))
        right_row.addWidget(btn_copy_r)
        dual_layout.addLayout(right_row)

        self.babble_url_dual.setVisible(False)
        grp_babble_layout.addWidget(self.babble_url_dual)

        self.lbl_mjpeg_status = QLabel("MJPEG Server: Off")
        self.lbl_mjpeg_status.setStyleSheet("color: #888;")
        grp_babble_layout.addWidget(self.lbl_mjpeg_status)

        self.grp_babble.setLayout(grp_babble_layout)
        self.grp_babble.setVisible(False)
        settings_panel.addWidget(self.grp_babble)

        # Smoothing Group
        grp_smooth = QGroupBox("Signal Smoothing")
        grp_smooth_layout = QVBoxLayout()
        self.lbl_smooth_val = QLabel("Smooth Intensity: 50%")
        grp_smooth_layout.addWidget(self.lbl_smooth_val)
        
        self.slider_smooth = QSlider(Qt.Horizontal)
        self.slider_smooth.setRange(0, 99)
        self.slider_smooth.setValue(50)
        self.slider_smooth.valueChanged.connect(lambda v: self.lbl_smooth_val.setText(f"Smooth Intensity: {v}%"))
        self.slider_smooth.valueChanged.connect(lambda v: self._update_setting("smooth", v))
        grp_smooth_layout.addWidget(self.slider_smooth)
        
        grp_smooth.setLayout(grp_smooth_layout)
        settings_panel.addWidget(grp_smooth)

        # Sync Group
        grp_sync = QGroupBox("L/R Output Matching")
        grp_sync_layout =QVBoxLayout()
        self.lbl_sync_val = QLabel("Sync: 0%")
        grp_sync_layout.addWidget(self.lbl_sync_val)
        
        self.slider_sync = QSlider(Qt.Horizontal)
        self.slider_sync.setRange(0, 100)
        self.slider_sync.setValue(0)
        self.slider_sync.valueChanged.connect(lambda v: self.lbl_sync_val.setText(f"Sync: {v}%"))
        self.slider_sync.valueChanged.connect(lambda v: self._update_setting("sync", v))
        grp_sync_layout.addWidget(self.slider_sync)

        self.btn_sym_calib = QPushButton("Auto Symmetry Match")
        self.btn_sym_calib.setToolTip(
            "Sample Neutral, Max Up, and Max Down for 2s each, then scale left/right "
            "output to match."
        )
        self.btn_sym_calib.clicked.connect(self.start_symmetry_calibration)
        grp_sync_layout.addWidget(self.btn_sym_calib)
        
        grp_sync.setLayout(grp_sync_layout)
        settings_panel.addWidget(grp_sync)
        
        # --- Headset Recenter ---
        auto_group = QGroupBox("Headset Recenter")
        auto_layout = QVBoxLayout()

        self.btn_set_neutral = QPushButton("Recenter Neutral")
        self.btn_set_neutral.setToolTip("Press while relaxed. Resets the zero point for both eyes.")
        self.btn_set_neutral.clicked.connect(self.set_neutral_baseline)
        auto_layout.addWidget(self.btn_set_neutral)

        self.chk_auto_baseline = QCheckBox("Auto-follow drift")
        self.chk_auto_baseline.setChecked(False)
        self.chk_auto_baseline.setToolTip("Slowly recenter when your face is still.")
        self.chk_auto_baseline.stateChanged.connect(self.toggle_auto_baseline)
        self.chk_auto_baseline.stateChanged.connect(lambda v: self._update_setting("auto_baseline", v == Qt.Checked))
        auto_layout.addWidget(self.chk_auto_baseline)

        # Hidden advanced controls (slider_alpha still functional, just not shown)
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(1, 100)
        self.slider_alpha.setValue(5)
        self.slider_alpha.setVisible(False)

        self.lbl_auto_status = QLabel("")
        self.lbl_auto_status.setProperty("class", "muted-label")
        auto_layout.addWidget(self.lbl_auto_status)

        auto_group.setLayout(auto_layout)
        settings_panel.addWidget(auto_group)
        # -------------------------------------
        
        settings_panel.addStretch()
        layout.addLayout(settings_panel, stretch=1)
        
    def toggle_auto_baseline(self, state):
        if state == 0:
            self.lbl_auto_status.setText("")
        else:
            self.lbl_auto_status.setText("Watching for drift...")
            
    def reset_auto_baseline(self):
        self.auto_offset_brow_l = 0.0
        self.auto_offset_brow_r = 0.0
        self.auto_offset_inner_l = 0.0
        self.auto_offset_inner_r = 0.0
        self.auto_offset_outer_l = 0.0
        self.auto_offset_outer_r = 0.0
        self._baseline_stamps.clear()
        self._baseline_locked = False
        self.shift_tracker_l.reset()
        self.shift_tracker_r.reset()
        self.lbl_auto_status.setText("Reset.")
        self.lbl_auto_status.setStyleSheet("color: #888;")

    def setup_calibration_tab(self):
        layout = QVBoxLayout(self.tab_calibration)
        lbl_capture = QLabel("Guided Dataset Capture")
        lbl_capture.setProperty("class", "bold-label")
        layout.addWidget(lbl_capture)
        
        # Automatic Guided Sequence UI
        seq_controls = QHBoxLayout()
        
        self.btn_start_seq = QPushButton("START GUIDED CAPTURE")
        self.btn_start_seq.setProperty("class", "primary-btn-success")
        self.btn_start_seq.setToolTip(
            "Record labeled eyebrow training frames for Neutral, Brows Up, Frown, Sad Inner, "
            "and Smile Outer, including random-gaze stages."
        )
        self.btn_start_seq.clicked.connect(self.start_calibration_sequence)
        seq_controls.addWidget(self.btn_start_seq)
        
        self.btn_stop_seq = QPushButton("STOP CAPTURE")
        self.btn_stop_seq.setProperty("class", "primary-btn-danger")
        self.btn_stop_seq.setToolTip(
            "Abort the current guided capture and discard frames recorded in this run."
        )
        self.btn_stop_seq.clicked.connect(self.stop_calibration_sequence)
        self.btn_stop_seq.setEnabled(False)
        seq_controls.addWidget(self.btn_stop_seq)
        
        seq_layout = QVBoxLayout()
        seq_layout.addLayout(seq_controls)
        
        self.lbl_seq_instruction = QLabel("Ready to capture training data")
        self.lbl_seq_instruction.setAlignment(Qt.AlignCenter)
        self.lbl_seq_instruction.setFont(QFont("Arial", 24, QFont.Bold))
        self.lbl_seq_instruction.setStyleSheet("color: #111; margin-top: 20px; margin-bottom: 20px;")
        seq_layout.addWidget(self.lbl_seq_instruction)
        
        layout.addLayout(seq_layout)
        
        # Dataset Management
        data_layout = QHBoxLayout()
        self.lbl_dataset_status = QLabel("  |  Captured training images: 0")
        self.lbl_dataset_status.setProperty("class", "muted-label")
        data_layout.addWidget(self.lbl_dataset_status)
        
        self.btn_clear_data = QPushButton("Clear Captured Data")
        self.btn_clear_data.setProperty("class", "primary-btn-danger")
        self.btn_clear_data.setToolTip(
            "Delete all captured eyebrow training images and the generated CSV files."
        )
        self.btn_clear_data.clicked.connect(self.clear_calibration_data)
        data_layout.addWidget(self.btn_clear_data)
        layout.addLayout(data_layout)
        
        # Training Section
        lbl_train = QLabel("Train / Bake Model")
        lbl_train.setFont(QFont("Arial", 12, QFont.Bold))
        lbl_train.setProperty("class", "header-label")
        layout.addWidget(lbl_train)

        train_btn_row = QHBoxLayout()
        # Setup button (one-time Python + PyTorch install)
        self.btn_setup_training = QPushButton("Setup Training Environment")
        self.btn_setup_training.setProperty("class", "primary-btn")
        self.btn_setup_training.setToolTip("One-time download: Python + PyTorch (~300MB)")
        self.btn_setup_training.clicked.connect(self._start_training_setup)
        if _get_training_python() is not None:
            self.btn_setup_training.setText("Training Environment Ready")
            self.btn_setup_training.setEnabled(False)
        layout.addWidget(self.btn_setup_training)

        self.btn_train = QPushButton("BAKE MODEL FROM CAPTURED DATA")
        self.btn_train.setObjectName("btn_bake_main")
        self.btn_train.setProperty("class", "primary-btn-purple")
        self.btn_train.clicked.connect(self.start_training)
        train_btn_row.addWidget(self.btn_train)

        self.btn_train_with_path = QPushButton("BAKE FROM OTHER FOLDER")
        self.btn_train_with_path.setObjectName("btn_bake_with_path")
        self.btn_train_with_path.setProperty("class", "primary-btn")
        self.btn_train_with_path.clicked.connect(self.start_training_with_path)
        train_btn_row.addWidget(self.btn_train_with_path)

        layout.addLayout(train_btn_row)

        self.lbl_train_status = QLabel("Status: Idle")
        layout.addWidget(self.lbl_train_status)

        # Embedded Console
        grp_console = QGroupBox("Console")
        console_layout = QVBoxLayout()
        self.txt_console = QPlainTextEdit()
        self.txt_console.setReadOnly(True)
        self.txt_console.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.txt_console.document().setMaximumBlockCount(2000)
        self.txt_console.setObjectName("console-text")
        self.txt_console.setMaximumHeight(200)
        mono = QFont("Consolas", 9)
        mono.setStyleHint(QFont.Monospace)
        self.txt_console.setFont(mono)
        console_layout.addWidget(self.txt_console)
        grp_console.setLayout(console_layout)
        layout.addWidget(grp_console)

        # Calibration State Machine
        # Recording states use frame count, REST states use seconds
        self.is_calibrating = False
        self.calib_states = [
            {"name": "REST (Prepare for Neutral)", "target": None, "duration": 3.0},
            {"name": "NEUTRAL (Resting)", "target": 0.0, "folder": "neutral_resting", "frames": 300},
            {"name": "NEUTRAL + Random Gaze", "target": 0.0, "folder": "neutral_random_gaze", "frames": 750},
            {"name": "REST (Prepare for Surprised)", "target": None, "duration": 3.0},
            {"name": "SURPRISED (Brows UP)", "target": 1.0, "folder": "surprised_brows_up", "frames": 300},
            {"name": "SURPRISED + Random Gaze", "target": 1.0, "folder": "surprised_brows_up_random_gaze", "frames": 750},
            {"name": "REST (Prepare for Lower Eyebrow)", "target": None, "duration": 3.0},
            {"name": "LOWER EYEBROW (Frown)", "target": -1.0, "folder": "frown_brows_down", "frames": 300},
            {"name": "FROWN + Random Gaze", "target": -1.0, "folder": "frown_brows_down_random_gaze", "frames": 750},
            {"name": "REST (Prepare for Sad)", "target": None, "duration": 3.0},
            {"name": "SAD (Inner Brows UP)", "target": 0.5, "folder": "sad_inner_brows_up", "frames": 300},
            {"name": "SAD INNER + Random Gaze", "target": 0.5, "folder": "sad_inner_brows_up_random_gaze", "frames": 750},
            {"name": "REST (Prepare for Smile)", "target": None, "duration": 3.0},
            {"name": "SMILE (Outer Brows DOWN)", "target": -0.5, "folder": "smile_outer_brows_down", "frames": 300},
            {"name": "SMILE OUTER + Random Gaze", "target": -0.5, "folder": "smile_outer_brows_down_random_gaze", "frames": 750}
        ]
        self.calib_idx = 0
        self.calib_start_time = 0.0
        self.calib_frame_count = 0

    def _setup_console(self):
        """Wire stdout/stderr to the embedded console widget."""
        self.log_emitter = LogEmitter()
        self.log_emitter.message.connect(self._append_log)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = StreamRedirect(self.log_emitter)
        sys.stderr = StreamRedirect(self.log_emitter)
        
    def snap_left_slider(self, value):
        if -15 < value < 15 and value != 0:
            self.slider_l.blockSignals(True)
            self.slider_l.setValue(0)
            self.slider_l.blockSignals(False)
            
    def snap_right_slider(self, value):
        if -15 < value < 15 and value != 0:
            self.slider_r.blockSignals(True)
            self.slider_r.setValue(0)
            self.slider_r.blockSignals(False)

    def setup_settings_tab(self):
        layout = QVBoxLayout(self.tab_settings)

        # Update / Token
        grp_update = QGroupBox("App Updates")
        update_layout = QHBoxLayout()
        self.lbl_version_settings = QLabel(f"v{APP_VERSION}")
        self.lbl_version_settings.setProperty("class", "muted-label")
        update_layout.addWidget(self.lbl_version_settings)
        self.btn_check_updates = QPushButton("Check for Updates")
        self.btn_check_updates.clicked.connect(self.check_for_updates)
        update_layout.addWidget(self.btn_check_updates)
        self.btn_set_token = QPushButton("Set GitHub Token")
        self.btn_set_token.clicked.connect(self.set_github_token)
        update_layout.addWidget(self.btn_set_token)
        update_layout.addStretch(1)
        grp_update.setLayout(update_layout)
        layout.addWidget(grp_update)

        # Compute Device
        grp_device = QGroupBox("Compute Device")
        grp_device_layout = QVBoxLayout()
        self.cmb_device = QComboBox()
        for label, _dev in self.available_devices:
            self.cmb_device.addItem(label)
        self.cmb_device.currentIndexChanged.connect(self.on_device_changed)
        grp_device_layout.addWidget(self.cmb_device)
        grp_device.setLayout(grp_device_layout)
        layout.addWidget(grp_device)

        # Appearance
        grp_theme = QGroupBox("Appearance")
        theme_layout = QHBoxLayout()
        self.btn_theme = QPushButton("Light Mode")
        self.btn_theme.clicked.connect(self.toggle_theme)
        theme_layout.addWidget(self.btn_theme)
        theme_layout.addStretch(1)
        grp_theme.setLayout(theme_layout)
        layout.addWidget(grp_theme)

        # HMD Profile
        grp_hmd = QGroupBox("HMD Profile")
        hmd_layout = QVBoxLayout()
        self.cmb_hmd = QComboBox()
        self.cmb_hmd.addItem("Pimax / Varjo")
        self.cmb_hmd.addItem("Bigscreen Beyond 2e")
        self.cmb_hmd.addItem("DIY")
        self.cmb_hmd.currentIndexChanged.connect(self._on_hmd_changed)
        hmd_layout.addWidget(self.cmb_hmd)
        grp_hmd.setLayout(hmd_layout)
        layout.addWidget(grp_hmd)

        # OSC
        grp_osc = QGroupBox("OSC Output")
        grp_osc_layout = QVBoxLayout()
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("IP:"))
        self.txt_ip = QLineEdit("127.0.0.1")
        self.txt_ip.textChanged.connect(lambda v: self._update_setting("osc_ip", v))
        port_layout.addWidget(self.txt_ip)
        port_layout.addWidget(QLabel("Port:"))
        self.txt_port = QLineEdit("9000")
        self.txt_port.setFixedWidth(60)
        self.txt_port.textChanged.connect(lambda v: self._update_setting("osc_port", v))
        port_layout.addWidget(self.txt_port)
        grp_osc_layout.addLayout(port_layout)
        self.btn_osc = QPushButton("Start OSC Sender")
        self.btn_osc.setProperty("class", "success-btn")
        self.btn_osc.clicked.connect(self.toggle_osc)
        grp_osc_layout.addWidget(self.btn_osc)
        grp_osc.setLayout(grp_osc_layout)
        layout.addWidget(grp_osc)

        # Baballonia Camera Sharing
        grp_babble_settings = QGroupBox("Baballonia Camera Sharing")
        babble_s_layout = QVBoxLayout()
        babble_s_layout.addWidget(QLabel("MJPEG port for sharing camera with Baballonia."))
        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Port:"))
        self.txt_babble_port_settings = QLineEdit(str(self.settings.get("baballonia_mjpeg_port", 8085)))
        self.txt_babble_port_settings.setFixedWidth(60)
        self.txt_babble_port_settings.textChanged.connect(
            lambda v: (self._update_setting("baballonia_mjpeg_port", int(v) if v.isdigit() else 8085),
                       setattr(self.txt_babble_port, 'text_pending', v))
        )
        port_row.addWidget(self.txt_babble_port_settings)
        port_row.addStretch(1)
        babble_s_layout.addLayout(port_row)
        grp_babble_settings.setLayout(babble_s_layout)
        layout.addWidget(grp_babble_settings)

        layout.addStretch()

    # --- Actions ---


    def apply_theme(self):
        dark_qss = """
        QMainWindow, QWidget { background-color: #1e1e1e; color: #f0f0f0; }
        QTabWidget::pane { border: 1px solid #333; border-radius: 4px; top:-1px; }
        QTabBar::tab { background: #2d2d2d; color: #aaa; padding: 8px 16px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
        QTabBar::tab:selected { background: #3d3d3d; color: white; font-weight: bold; border: 1px solid #444; border-bottom: none; }
        QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 12px; font-weight: bold; padding-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #aaa; }
        QLineEdit { background: #333; color: white; border: 1px solid #555; border-radius: 3px; padding: 4px; }
        QProgressBar { background: #333; border: 1px solid #444; border-radius: 6px; text-align: center; }
        QProgressBar::chunk { border-radius: 4px; }
        QProgressBar[class="bar-red"]::chunk { background-color: #f44336; }
        QProgressBar[class="bar-green"]::chunk { background-color: #4CAF50; }
        QLabel { background-color: transparent; }
        QPushButton { background-color: #333; color: white; border: 1px solid #555; border-radius: 4px; padding: 6px; }
        QPushButton:hover { background-color: #444; }
        
        QPushButton[class="primary-btn"] { background-color: #0078D7; color: white; font-weight: bold; padding: 10px; border: none; font-size: 14px;}
        QPushButton[class="primary-btn"]:hover { background-color: #005A9E; }
        QPushButton[class="primary-btn-success"] { background-color: #2e7d32; color: white; font-weight: bold; padding: 10px; border: none; font-size: 14px;}
        QPushButton[class="primary-btn-success"]:hover { background-color: #1b5e20; }
        QPushButton[class="primary-btn-danger"] { background-color: #d32f2f; color: white; font-weight: bold; padding: 10px; border: none; font-size: 14px;}
        QPushButton[class="primary-btn-danger"]:hover { background-color: #b71c1c; }
        QPushButton[class="primary-btn-purple"] { background-color: #673AB7; color: white; font-weight: bold; padding: 15px; border: none; font-size: 14px;}
        QPushButton[class="primary-btn-purple"]:hover { background-color: #512DA8; }
        
        QPushButton[class="success-btn"] { background-color: #2e7d32; color: white; font-weight: bold; padding: 8px; border: none; }
        QPushButton[class="success-btn"]:hover { background-color: #1b5e20; }
        QPushButton[class="danger-btn"] { background-color: #d32f2f; color: white; font-weight: bold; padding: 8px; border: none; }
        QPushButton[class="danger-btn"]:hover { background-color: #b71c1c; }
        
        QPushButton[class="theme-btn"] { background-color: transparent; border: 1px solid #555; border-radius: 12px; padding: 4px 12px;}
        QPushButton[class="theme-btn"]:hover { background-color: #333; }
        
        QLabel[class="cam-label"] { background-color: #333333; border: 2px solid #444; border-radius: 8px; }
        QLabel[class="muted-label"] { color: #888; font-size: 11px; }
        QLabel[class="bold-label"] { color: #ccc; font-size: 13px; font-weight: bold; }
        QLabel[class="header-label"] { color: #ddd; font-size: 14px; font-weight: bold; margin-top: 20px; }
        QCheckBox[class="tiny-check"] { font-size: 11px; padding: 2px 4px; }

        QPushButton#btn_bake_main, QPushButton#btn_bake_with_path { padding: 15px; font-size: 14px; }
        
        QScrollArea { border: none; }
        QScrollBar:vertical { background: #1e1e1e; width: 12px; margin: 0px; }
        QScrollBar::handle:vertical { background: #444; min-height: 20px; border-radius: 6px; margin: 2px;}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QPlainTextEdit#console-text, QTextEdit#console-text {
            background: #101214;
            color: #e6e6e6;
            border: 1px solid #2b2f33;
            border-radius: 6px;
            padding: 8px;
            selection-background-color: #2b4a7f;
        }
        """
        
        light_qss = """
        QMainWindow, QWidget { background-color: #f5f5f5; color: #111; }
        QTabWidget::pane { border: 1px solid #ccc; border-radius: 4px; background: white; top:-1px; }
        QTabBar::tab { background: #e0e0e0; color: #555; padding: 8px 16px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
        QTabBar::tab:selected { background: white; color: #111; font-weight: bold; border: 1px solid #ccc; border-bottom: none; }
        QGroupBox { border: 1px solid #bbb; border-radius: 6px; margin-top: 12px; font-weight: bold; padding-top: 10px; background: white; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #444; }
        QLineEdit { background: #fff; color: black; border: 1px solid #aaa; border-radius: 3px; padding: 4px; }
        QProgressBar { background: #e0e0e0; border: 1px solid #ccc; border-radius: 6px; text-align: center; }
        QProgressBar::chunk { border-radius: 4px; }
        QProgressBar[class="bar-red"]::chunk { background-color: #d32f2f; }
        QProgressBar[class="bar-green"]::chunk { background-color: #2e7d32; }
        QLabel { background-color: transparent; }
        QPushButton { background-color: #e0e0e0; color: black; border: 1px solid #bbb; border-radius: 4px; padding: 6px; }
        QPushButton:hover { background-color: #d0d0d0; }
        
        QPushButton[class="primary-btn"] { background-color: #0078D7; color: white; font-weight: bold; padding: 10px; border: none; font-size: 14px;}
        QPushButton[class="primary-btn"]:hover { background-color: #005A9E; }
        QPushButton[class="primary-btn-success"] { background-color: #2e7d32; color: white; font-weight: bold; padding: 10px; border: none; font-size: 14px;}
        QPushButton[class="primary-btn-success"]:hover { background-color: #1b5e20; }
        QPushButton[class="primary-btn-danger"] { background-color: #d32f2f; color: white; font-weight: bold; padding: 10px; border: none; font-size: 14px;}
        QPushButton[class="primary-btn-danger"]:hover { background-color: #b71c1c; }
        QPushButton[class="primary-btn-purple"] { background-color: #673AB7; color: white; font-weight: bold; padding: 15px; border: none; font-size: 14px;}
        QPushButton[class="primary-btn-purple"]:hover { background-color: #512DA8; }
        
        QPushButton[class="success-btn"] { background-color: #2e7d32; color: white; font-weight: bold; padding: 8px; border: none; }
        QPushButton[class="success-btn"]:hover { background-color: #1b5e20; }
        QPushButton[class="danger-btn"] { background-color: #d32f2f; color: white; font-weight: bold; padding: 8px; border: none; }
        QPushButton[class="danger-btn"]:hover { background-color: #b71c1c; }
        
        QPushButton[class="theme-btn"] { background-color: transparent; border: 1px solid #bbb; border-radius: 12px; padding: 4px 12px;}
        QPushButton[class="theme-btn"]:hover { background-color: #e0e0e0; }
        
        QLabel[class="cam-label"] { background-color: #dedede; border: 2px solid #ccc; border-radius: 8px; }
        QLabel[class="muted-label"] { color: #555; font-size: 11px; }
        QLabel[class="bold-label"] { color: #333; font-size: 13px; font-weight: bold; }
        QLabel[class="header-label"] { color: #222; font-size: 14px; font-weight: bold; margin-top: 20px; }
        QCheckBox[class="tiny-check"] { font-size: 11px; padding: 2px 4px; }

        QPushButton#btn_bake_main, QPushButton#btn_bake_with_path { padding: 15px; font-size: 14px; }
        
        QScrollArea { border: none; background: transparent; }
        QScrollBar:vertical { background: #f5f5f5; width: 12px; margin: 0px; }
        QScrollBar::handle:vertical { background: #ccc; min-height: 20px; border-radius: 6px; margin: 2px;}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QPlainTextEdit#console-text, QTextEdit#console-text {
            background: #fbfbfb;
            color: #1a1a1a;
            border: 1px solid #cfcfcf;
            border-radius: 6px;
            padding: 8px;
            selection-background-color: #bcd6ff;
        }
        """
        
        if self.is_dark_mode:
            self.setStyleSheet(dark_qss)
            self.btn_theme.setText("☀️ Light Mode")
        else:
            self.setStyleSheet(light_qss)
            self.btn_theme.setText("🌙 Dark Mode")
            
    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme()

    def set_neutral_baseline(self):
        # Capture current raw EMA values to zero them out + reset shift tracking origin
        try:
            if self.ema_left.value is None or self.ema_right.value is None:
                self._show_error_dialog("Warning", "No tracking data yet.\nStart the streams and wait for values before setting baseline.")
                return
            self.offset_l = self.ema_left.value
            self.offset_r = self.ema_right.value
            # Re-anchor shift trackers to current HMD position
            self.shift_tracker_l.reset()
            self.shift_tracker_r.reset()
            QMessageBox.information(self, "Recalibrated", f"New Neutral Offsets:\nLeft: {self.offset_l:.2f}\nRight: {self.offset_r:.2f}")
        except Exception as e:
            self._show_error_dialog("Error", f"Failed to set baseline:\n{e}")

    def reset_offsets_zero(self):
        self.offset_l = 0.0
        self.offset_r = 0.0
        self.auto_offset_brow_l = 0.0
        self.auto_offset_brow_r = 0.0
        self.auto_offset_inner_l = 0.0
        self.auto_offset_inner_r = 0.0
        self.auto_offset_outer_l = 0.0
        self.auto_offset_outer_r = 0.0

    def start_symmetry_calibration(self):
        if self.sym_calibrating:
            return
        if self.last_raw_brow_l is None or self.last_raw_brow_r is None:
            self._show_error_dialog("Warning", "No tracking data yet.\nStart the streams and wait for values before symmetry calibration.")
            return
        self.sym_calibrating = True
        self.sym_phase_idx = 0
        self.sym_phase_results = {}
        self.sym_samples_l = []
        self.sym_samples_r = []
        self.sym_phase_start = time.time()
        if hasattr(self, "btn_sym_calib"):
            self.btn_sym_calib.setEnabled(False)
        phase_name, duration = self.sym_phases[self.sym_phase_idx]
        self.lbl_auto_status.setText(f"Symmetry Calib: {phase_name} ({duration:.1f}s)")
        self.lbl_auto_status.setStyleSheet("color: #eb9534;")
        self.sym_timer.start(100)

    def _tick_symmetry_calibration(self):
        if not self.sym_calibrating:
            return
        if self.last_raw_brow_l is None or self.last_raw_brow_r is None:
            return
        self.sym_samples_l.append(self.last_raw_brow_l)
        self.sym_samples_r.append(self.last_raw_brow_r)
        phase_name, duration = self.sym_phases[self.sym_phase_idx]
        if (time.time() - self.sym_phase_start) >= duration:
            if len(self.sym_samples_l) == 0 or len(self.sym_samples_r) == 0:
                self._finish_symmetry_calibration(error="No samples collected. Try again.")
                return
            mean_l = sum(self.sym_samples_l) / len(self.sym_samples_l)
            mean_r = sum(self.sym_samples_r) / len(self.sym_samples_r)
            self.sym_phase_results[phase_name] = (mean_l, mean_r)
            self.sym_phase_idx += 1
            if self.sym_phase_idx >= len(self.sym_phases):
                self._finish_symmetry_calibration()
                return
            self.sym_samples_l = []
            self.sym_samples_r = []
            self.sym_phase_start = time.time()
            next_name, next_dur = self.sym_phases[self.sym_phase_idx]
            self.lbl_auto_status.setText(f"Symmetry Calib: {next_name} ({next_dur:.1f}s)")

    def _finish_symmetry_calibration(self, error=None):
        self.sym_timer.stop()
        self.sym_calibrating = False
        if hasattr(self, "btn_sym_calib"):
            self.btn_sym_calib.setEnabled(True)
        if error:
            self.lbl_auto_status.setText(f"Symmetry Calib: {error}")
            self.lbl_auto_status.setStyleSheet("color: #d9534f;")
            return
        needed = {"Neutral", "Max Up", "Max Down"}
        if not needed.issubset(self.sym_phase_results.keys()):
            self.lbl_auto_status.setText("Symmetry Calib: Missing phases. Try again.")
            self.lbl_auto_status.setStyleSheet("color: #d9534f;")
            return
        n_l, n_r = self.sym_phase_results["Neutral"]
        up_l, up_r = self.sym_phase_results["Max Up"]
        down_l, down_r = self.sym_phase_results["Max Down"]

        def _range(neutral, up, down):
            up_delta = up - neutral
            down_delta = neutral - down
            return max(abs(up_delta), abs(down_delta))

        range_l = _range(n_l, up_l, down_l)
        range_r = _range(n_r, up_r, down_r)
        if range_l < 0.02 or range_r < 0.02:
            self.lbl_auto_status.setText("Symmetry Calib: Not enough movement. Try again.")
            self.lbl_auto_status.setStyleSheet("color: #d9534f;")
            return
        avg_range = (range_l + range_r) / 2.0
        eps = 1e-6
        scale_l = avg_range / max(range_l, eps)
        scale_r = avg_range / max(range_r, eps)
        scale_l = max(0.33, min(3.0, scale_l))
        scale_r = max(0.33, min(3.0, scale_r))

        self.sym_offset_l = n_l
        self.sym_offset_r = n_r
        self.sym_scale_l = scale_l
        self.sym_scale_r = scale_r
        self._update_setting("sym_offset_l", self.sym_offset_l)
        self._update_setting("sym_offset_r", self.sym_offset_r)
        self._update_setting("sym_scale_l", self.sym_scale_l)
        self._update_setting("sym_scale_r", self.sym_scale_r)

        self.lbl_auto_status.setText(f"Symmetry Match saved: L x{scale_l:.2f}, R x{scale_r:.2f}")
        self.lbl_auto_status.setStyleSheet("color: #4CAF50;")
        self._baseline_stamps.clear()

    def update_dataset_status(self):
        self.lbl_dataset_status.setText(f"  |  Captured training images: {len(self.recorded_frames)}")

    def prune_dataset(self, records, root_dir, key_func, limit):
        # Class-Balanced Pruning: If over limit, delete oldest images from the largest class bucket
        pruned = False
        while len(records) > limit:
            pruned = True
            # Build buckets
            buckets = {}
            for i, r in enumerate(records):
                k = key_func(r)
                if k not in buckets: buckets[k] = []
                buckets[k].append(i)
                
            # Find the largest bucket
            largest_bucket_key = max(buckets.keys(), key=lambda k: len(buckets[k]))
            idx_to_remove = buckets[largest_bucket_key][0] # Oldest is first
            
            # Delete file
            target_record = records[idx_to_remove]
            file_path = root_dir / target_record['filename']
            try:
                if file_path.exists(): file_path.unlink()
            except: pass
            
            # Remove from list
            records.pop(idx_to_remove)
            
        return pruned

    def clear_calibration_data(self):
        if QMessageBox.question(self, "Confirm", "Are you sure you want to delete all captured eyebrow training images and CSVs?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.recorded_frames.clear()
            errors = []
            try:
                if self.csv_path.exists(): self.csv_path.unlink()
            except Exception as e:
                errors.append(f"{self.csv_path}: {e}")
            try:
                if self.val_csv_path.exists(): self.val_csv_path.unlink()
            except Exception as e:
                errors.append(f"{self.val_csv_path}: {e}")
            for f in self.eyebrow_images_dir.rglob("*.jpg"):
                try:
                    f.unlink()
                except Exception as e:
                    errors.append(f"{f}: {e}")
            self.update_dataset_status()
            QMessageBox.information(self, "Cleared", "Captured eyebrow training data cleared.")
            if errors:
                msg = "Some files could not be deleted:\n" + "\n".join(errors[:10])
                if len(errors) > 10:
                    msg += f"\n...and {len(errors) - 10} more."
                self._show_error_dialog("Warning", msg)

    def _update_connection_state(self):
        self.is_connected = self.is_connected_left or self.is_connected_right
        if self.is_connected:
            self.timer.start(10) # 100hz loop to allow 60fps pacing check to trigger precisely
        else:
            self.timer.stop()

    def toggle_left_connection(self):
        if not self.is_connected_left:
            if self.use_combined_feed and self.cmb_cam_l.currentData() == "url":
                QMessageBox.warning(self, "Camera Error", "Select a camera source before starting the stream.")
                return
            left_source = self._get_camera_source(self.cmb_cam_l, self.txt_cam_l.text())
            if isinstance(left_source, int) and left_source not in self.camera_devices:
                QMessageBox.warning(self, "Camera Error", f"Left camera index {left_source} not available. Please rescan.")
                return
            self.cam_left = CameraThread(left_source)
            self.cam_left.start()
            self.is_connected_left = True
            self.btn_connect_left.setText("Stop Stream" if self.use_combined_feed else "Stop Left Stream")
            self.btn_connect_left.setProperty("class", "primary-btn-danger"); self.btn_connect_left.style().unpolish(self.btn_connect_left); self.btn_connect_left.style().polish(self.btn_connect_left)
        else:
            self.is_connected_left = False
            self.btn_connect_left.setText("Start Stream" if self.use_combined_feed else "Start Left Stream")
            self.btn_connect_left.setProperty("class", "primary-btn"); self.btn_connect_left.style().unpolish(self.btn_connect_left); self.btn_connect_left.style().polish(self.btn_connect_left)
            if self.cam_left: self.cam_left.stop()
            self.left_img_label.clear()
            self.left_img_label.setText("Left Eye Stream")
        self._update_connection_state()

    def toggle_right_connection(self):
        if not self.is_connected_right:
            right_source = self._get_camera_source(self.cmb_cam_r, self.txt_cam_r.text())
            if isinstance(right_source, int) and right_source not in self.camera_devices:
                QMessageBox.warning(self, "Camera Error", f"Right camera index {right_source} not available. Please rescan.")
                return
            self.cam_right = CameraThread(right_source)
            self.cam_right.start()
            self.is_connected_right = True
            self.btn_connect_right.setText("Stop Right Stream")
            self.btn_connect_right.setProperty("class", "primary-btn-danger"); self.btn_connect_right.style().unpolish(self.btn_connect_right); self.btn_connect_right.style().polish(self.btn_connect_right)
        else:
            self.is_connected_right = False
            self.btn_connect_right.setText("Start Right Stream")
            self.btn_connect_right.setProperty("class", "primary-btn"); self.btn_connect_right.style().unpolish(self.btn_connect_right); self.btn_connect_right.style().polish(self.btn_connect_right)
            if self.cam_right: self.cam_right.stop()
            self.right_img_label.clear()
            self.right_img_label.setText("Right Eye Stream")
        self._update_connection_state()

    def toggle_osc(self):
        if not self.osc_enabled:
            # Connect
            ip = self.txt_ip.text().strip()
            try:
                port = int(self.txt_port.text())
                if port < 1 or port > 65535:
                    raise ValueError("Port must be between 1 and 65535.")
            except Exception as e:
                self._show_error_dialog("OSC Error", f"Invalid port:\n{e}")
                return
            try:
                self.osc_client = SimpleUDPClient(ip, port)
                
                self.osc_enabled = True
                for btn in (self.btn_osc, self.btn_osc_main):
                    btn.setText("Stop OSC Sender")
                    btn.setProperty("class", "danger-btn")
                    self._refresh_button_style(btn)
                self.txt_ip.setEnabled(False)
                self.txt_port.setEnabled(False)
            except Exception as e:
                QMessageBox.warning(self, "OSC Error", f"Could not create OSC client:\n{e}")
        else:
            # Disconnect
            self.osc_enabled = False
            self.osc_client = None
            
            for btn in (self.btn_osc, self.btn_osc_main):
                btn.setText("Start OSC Sender")
                btn.setProperty("class", "success-btn")
                self._refresh_button_style(btn)
            self.txt_ip.setEnabled(True)
            self.txt_port.setEnabled(True)

    def browse_weights(self):
        options = QFileDialog.Options()
        start_dir = self._get_model_dir()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Eyebrow Model", start_dir, "Models (*.onnx *.pth *.pt);;All Files (*)", options=options)
        if file_name:
            success = self.load_weights(file_name)
            if success:
                self.lbl_current_model.setText(os.path.basename(self.current_model_path))
                self._update_setting("last_model_path", self.current_model_path)
                QMessageBox.information(self, "Success", "Eyebrow model loaded successfully.")
            else:
                QMessageBox.critical(self, "Error", "Failed to load model.")
                
    def _map_folder_to_targets(self, folder, default_brow=0.0):
        name = (folder or "").lower()
        if "surprised" in name:
            return 1.0, 1.0, 1.0
        if "frown" in name:
            return -1.0, -1.0, 0.0
        if "sad_inner" in name:
            return 0.5, 1.0, 0.0
        if "smile_outer" in name:
            return -0.5, 0.0, -1.0
        if "neutral" in name:
            return 0.0, 0.0, 0.0
        return float(default_brow), 0.0, 0.0

    def save_calibration_frame(self, target_val, label_str, frame_l, frame_r):
        try:
            uid = int(time.time() * 1000)
            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            
            folder_dir = self.eyebrow_images_dir / label_str
            folder_dir.mkdir(parents=True, exist_ok=True)
            
            brow_t, inner_t, outer_t = self._map_folder_to_targets(label_str, target_val)

            name_l = f"{label_str}/{label_str}_l_{uid}.jpg"
            cv2.imwrite(str(self.eyebrow_images_dir / name_l), gray_l)
            self.recorded_frames.append({"filename": name_l, "brow": brow_t, "inner": inner_t, "outer": outer_t})
            
            name_r = f"{label_str}/{label_str}_r_{uid}.jpg"
            cv2.imwrite(str(self.eyebrow_images_dir / name_r), cv2.flip(gray_r, 1))
            self.recorded_frames.append({"filename": name_r, "brow": brow_t, "inner": inner_t, "outer": outer_t})
            
            limit = None
            was_pruned = False
            
            df = pd.DataFrame(self.recorded_frames)
            df = df.sample(frac=1).reset_index(drop=True)
            train_size = int(len(df) * 0.8)
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.iloc[:train_size].to_csv(self.csv_path, index=False)
            df.iloc[train_size:].to_csv(self.val_csv_path, index=False)
            self.update_dataset_status()
        except Exception as e:
            self._show_error_dialog("Capture Error", f"Failed to save calibration frame:\n{e}", key="save_frame")

    def start_calibration_sequence(self):
        if self.use_combined_feed:
            if not self.is_connected_left and not self.is_connected_right:
                QMessageBox.warning(self, "Warning", "Please connect the combined camera stream first!")
                return
        else:
            if not (self.is_connected_left and self.is_connected_right):
                QMessageBox.warning(self, "Warning", "Please connect both camera streams first!")
                return
            
        if self.use_combined_feed:
            if (self.cam_left is None and self.cam_right is None):
                QMessageBox.warning(self, "Warning", "No camera frames received! Please check your eye tracking stream.")
                return
        else:
            if self.cam_left is None or self.cam_right is None or getattr(self.cam_left, 'latest_frame', None) is None or getattr(self.cam_right, 'latest_frame', None) is None:
                QMessageBox.warning(self, "Warning", "No camera frames received! Please check your eye tracking streams.")
                return
            
        capture_states = [s for s in self.calib_states if s['target'] is not None]
        msg_lines = [
            "This guided capture records labeled training images for model baking.",
            "It does not change the live tracker until you bake and load a model.",
            "",
            "Active recording stages:"
        ]
        msg_lines.extend(f"- {s['name']}: {s.get('frames', 300)} frames" for s in capture_states)
        msg_lines.extend([
            "",
            "Random-gaze stages help the model ignore eye direction.",
            "Make sure your headset is firmly positioned before you start."
        ])
        msg = "\n".join(msg_lines)
        
        reply = QMessageBox.information(self, "Calibration Sequence", msg, QMessageBox.Ok | QMessageBox.Cancel)
        if reply == QMessageBox.Cancel:
            return

        self.btn_start_seq.setEnabled(False)
        self.btn_stop_seq.setEnabled(True)
        self.is_calibrating = True
        self.calib_idx = 0
        self.calib_frame_count = 0
        self.calib_start_time = time.time()
        self.calib_start_count = len(self.recorded_frames)
        self.lbl_seq_instruction.setText(f"Get Ready: {self.calib_states[0]['name']}...")

    def _reset_seq_text(self):
        if not self.is_calibrating:
            self.lbl_seq_instruction.setText("Ready to capture training data")
            self.lbl_seq_instruction.setStyleSheet("color: #111; margin-top: 20px; margin-bottom: 20px;")

    def stop_calibration_sequence(self):
        self.is_calibrating = False
        self.btn_start_seq.setEnabled(True)
        self.btn_stop_seq.setEnabled(False)
        self.lbl_seq_instruction.setText("Capture stopped. Discarding partial data...")
        self.lbl_seq_instruction.setStyleSheet("color: #eb9534; margin-top: 20px; margin-bottom: 20px;")
        QTimer.singleShot(3000, self._reset_seq_text)
        
        # Rollback logic: Delete any images captured during this aborted session
        if hasattr(self, 'calib_start_count'):
            removed_count = 0
            while len(self.recorded_frames) > self.calib_start_count:
                target_record = self.recorded_frames.pop()
                file_path = self.eyebrow_images_dir / target_record['filename']
                try:
                    if file_path.exists(): file_path.unlink()
                    removed_count += 1
                except: pass
                
            if removed_count > 0:
                # Rewrite CSVs to match the rolled-back list
                df = pd.DataFrame(self.recorded_frames).sample(frac=1).reset_index(drop=True) if len(self.recorded_frames) > 0 else pd.DataFrame()
                if not df.empty:
                    t_sz = int(len(df) * 0.8)
                    df.iloc[:t_sz].to_csv(self.csv_path, index=False)
                    df.iloc[t_sz:].to_csv(self.val_csv_path, index=False)
                else:
                    if self.csv_path.exists(): self.csv_path.unlink()
                    if self.val_csv_path.exists(): self.val_csv_path.unlink()
                self.update_dataset_status()
                print(f"Aborted calibration. Rolled back {removed_count} images.")

    def update_frame(self):
        # We handle Manual mode even if offline
        offline_manual = not self.is_connected and self.chk_manual.isChecked() and self.tabs.currentIndex() == 0

        if not self.is_connected and not offline_manual:
            return
            
        curr_time = time.time()
        is_new_frame = False
        frame_l_bgr, frame_r_bgr = None, None
            
        if self.is_connected:
            frame_l_bgr = getattr(self.cam_left, 'latest_frame', None) if self.is_connected_left else None
            frame_r_bgr = getattr(self.cam_right, 'latest_frame', None) if self.is_connected_right else None

            if self.use_combined_feed:
                combined = frame_l_bgr if frame_l_bgr is not None else frame_r_bgr
                combined = self._apply_combined_transform(combined)
                # Share raw combined frame to Baballonia before splitting
                if self.mjpeg_sharing_enabled and combined is not None:
                    self.mjpeg_server.update_frame(combined.copy())
                frame_l_bgr, frame_r_bgr = self._split_combined_frame(combined)
            elif self.mjpeg_sharing_enabled:
                # Non-combined (DIY): share left and right separately
                if frame_l_bgr is not None:
                    self.mjpeg_server.update_frame_left(frame_l_bgr.copy())
                if frame_r_bgr is not None:
                    self.mjpeg_server.update_frame_right(frame_r_bgr.copy())
            
            if self.tabs.currentIndex() == 0:
                if (self.is_connected_left or self.use_combined_feed) and frame_l_bgr is None:
                    self.left_img_label.setText("Left: No Signal")
                if (self.is_connected_right or self.use_combined_feed) and frame_r_bgr is None:
                    self.right_img_label.setText("Right: No Signal")
            
            # Use `is` because cv2.read returns a new numpy array object every frame
            if getattr(self, 'last_frame_l', None) is frame_l_bgr and getattr(self, 'last_frame_r', None) is frame_r_bgr:
                is_new_frame = False
            else:
                is_new_frame = True
            
            self.last_frame_l = frame_l_bgr
            self.last_frame_r = frame_r_bgr

        # Automatic Calibration Logic
        if self.tabs.currentIndex() == 1 and self.is_calibrating and self.is_connected:
            state = self.calib_states[self.calib_idx]

            if state['target'] is None:
                # REST state: time-based
                elapsed = curr_time - self.calib_start_time
                seconds_left = max(0, int(state['duration'] - elapsed) + 1)
                self.lbl_seq_instruction.setStyleSheet("color: #eb9534; margin-top: 20px; margin-bottom: 20px;")
                self.lbl_seq_instruction.setText(f"{state['name']}\n... {seconds_left}s ...")
                done = elapsed >= state['duration']
            else:
                # Recording state: frame-based
                target_frames = state.get('frames', 300)
                remaining = target_frames - self.calib_frame_count
                self.lbl_seq_instruction.setStyleSheet("color: #d32f2f; margin-top: 20px; margin-bottom: 20px;")
                self.lbl_seq_instruction.setText(f"HOLD: {state['name']}\n{self.calib_frame_count} / {target_frames} frames")

                if is_new_frame and frame_l_bgr is not None and frame_r_bgr is not None:
                    folder_name = state.get('folder', 'unknown_folder')
                    self.save_calibration_frame(state['target'], folder_name, frame_l_bgr, frame_r_bgr)
                    self.calib_frame_count += 1
                done = self.calib_frame_count >= target_frames

            if done:
                self.calib_idx += 1
                self.calib_frame_count = 0
                if self.calib_idx >= len(self.calib_states):
                    self.is_calibrating = False
                    self.lbl_seq_instruction.setText("Dataset capture complete.\nYou can now bake the model.")
                    self.btn_start_seq.setEnabled(True)
                    self.btn_stop_seq.setEnabled(False)
                else:
                    self.calib_start_time = time.time()

        if self.is_connected:
            if frame_l_bgr is None or frame_r_bgr is None:
                # Allow single-stream display, but skip inference if either side is missing
                if self.tabs.currentIndex() == 0:
                    if frame_l_bgr is not None:
                        try:
                            gray_l = cv2.cvtColor(frame_l_bgr, cv2.COLOR_BGR2GRAY)
                            h, w = gray_l.shape
                            self.left_img_label.setPixmap(QPixmap.fromImage(QImage(gray_l.data, w, h, w, QImage.Format_Grayscale8)).scaled(self.left_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                        except Exception:
                            pass
                    if frame_r_bgr is not None:
                        try:
                            gray_r = cv2.cvtColor(frame_r_bgr, cv2.COLOR_BGR2GRAY)
                            h, w = gray_r.shape
                            self.right_img_label.setPixmap(QPixmap.fromImage(QImage(gray_r.data, w, h, w, QImage.Format_Grayscale8)).scaled(self.right_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                        except Exception:
                            pass
                return
            # If no new camera frame, skip inference
            if not is_new_frame:
                if not self.chk_manual.isChecked():
                    return
            
            gray_l, gray_r = None, None
            try:
                # Convert BGR OpenCV frames to grayscale
                gray_l = cv2.cvtColor(frame_l_bgr, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r_bgr, cv2.COLOR_BGR2GRAY)

                frame_l = frame_l_bgr
                frame_r = frame_r_bgr
            except Exception as e:
                print(f"Frame Conversion Error: {e}")
                return
            
            dt = curr_time - getattr(self, 'last_update_time', curr_time)
            self.last_update_time = curr_time
            if dt > 0:
                self.current_fps = self.current_fps * 0.9 + (1.0 / dt) * 0.1
        else:
            frame_l, frame_r = None, None

        # Inference Logic
        if self.is_connected and frame_l is not None and frame_r is not None:
            if self.tabs.currentIndex() == 0:
                # Displays (Always update visuals)
                h, w = gray_l.shape
                
                fps_l = int(getattr(self.cam_left, "fps", 0.0)) if self.is_connected_left or self.use_combined_feed else 0
                fps_r = int(getattr(self.cam_right, "fps", 0.0)) if self.is_connected_right else 0
                self.lbl_l_fps.setText(f"FPS: {fps_l}")
                self.lbl_r_fps.setText(f"FPS: {fps_r}")
                
                self.left_img_label.setPixmap(QPixmap.fromImage(QImage(gray_l.data, w, h, w, QImage.Format_Grayscale8)).scaled(self.left_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.right_img_label.setPixmap(QPixmap.fromImage(QImage(gray_r.data, w, h, w, QImage.Format_Grayscale8)).scaled(self.right_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # Decide where the values come from
            if self.chk_manual.isChecked():
                # Read from UI sliders (-100 to 100 -> -1.0 to 1.0)
                out_l = self.slider_l.value() / 100.0
                out_r = self.slider_r.value() / 100.0
                inner_l = out_l
                inner_r = out_r
                outer_l = out_l
                outer_r = out_r
                lbl_suffix = " (MANUAL)"
            elif self.is_connected and gray_l is not None and gray_r is not None:
                try:
                    # HMD shift detection (image stabilization)
                    shift_l = self.shift_tracker_l.update(gray_l)
                    shift_r = self.shift_tracker_r.update(gray_r)

                    # ONNX Runtime inference with shift-compensated crop
                    if self.model is None:
                        raise RuntimeError("Model not loaded")
                    out_l_vals, out_r_vals = self.model.predict_pair(gray_l, gray_r, shift_l=shift_l, shift_r=shift_r)

                    raw_brow_l, raw_inner_l, raw_outer_l = out_l_vals
                    raw_brow_r, raw_inner_r, raw_outer_r = out_r_vals

                    # If loaded weights are legacy (no inner/outer), mirror brow outputs
                    if not self.model_has_inner_outer:
                        raw_inner_l = raw_brow_l
                        raw_outer_l = raw_brow_l
                        raw_inner_r = raw_brow_r
                        raw_outer_r = raw_brow_r
                                
                    # --- AUTO BASELINE CORRECTION (time-based, all channels, FPS-independent) ---
                    now = time.time()
                    self._baseline_stamps.append((now, raw_brow_l, raw_brow_r,
                                                  raw_inner_l, raw_inner_r,
                                                  raw_outer_l, raw_outer_r))
                    # Trim to time window
                    cutoff = now - self._baseline_window_sec
                    while self._baseline_stamps and self._baseline_stamps[0][0] < cutoff:
                        self._baseline_stamps.pop(0)

                    if self.chk_auto_baseline.isChecked() and len(self._baseline_stamps) >= 10:
                        n = len(self._baseline_stamps)
                        # Compute variance of brow channels only (main signal for stability detection)
                        brow_l_vals = [s[1] for s in self._baseline_stamps]
                        brow_r_vals = [s[2] for s in self._baseline_stamps]
                        mean_bl = sum(brow_l_vals) / n
                        mean_br = sum(brow_r_vals) / n
                        var_l = sum((v - mean_bl) ** 2 for v in brow_l_vals) / (n - 1)
                        var_r = sum((v - mean_br) ** 2 for v in brow_r_vals) / (n - 1)
                        var_max = max(var_l, var_r)

                        # Hysteresis: different thresholds for lock/unlock
                        LOCK_THRESH = 0.0005
                        UNLOCK_THRESH = 0.0015
                        if self._baseline_locked:
                            is_stable = var_max < UNLOCK_THRESH
                        else:
                            is_stable = var_max < LOCK_THRESH
                        self._baseline_locked = is_stable

                        if is_stable:
                            # Frame-rate independent lerp: t = 1 - (1-speed)^dt
                            dt = now - self._baseline_stamps[-2][0] if n >= 2 else 0.016
                            speed = (self.slider_alpha.value() / 100.0) * 3.0  # per-second rate
                            t = 1.0 - (1.0 - min(speed, 0.99)) ** dt
                            stability = 1.0 - (var_max / UNLOCK_THRESH)
                            stability = max(0.0, min(1.0, stability))
                            t *= stability

                            # Compute mean of all 6 channels over the stable window
                            mean_il = sum(s[3] for s in self._baseline_stamps) / n
                            mean_ir = sum(s[4] for s in self._baseline_stamps) / n
                            mean_ol = sum(s[5] for s in self._baseline_stamps) / n
                            mean_or = sum(s[6] for s in self._baseline_stamps) / n

                            self.auto_offset_brow_l += (mean_bl - self.auto_offset_brow_l) * t
                            self.auto_offset_brow_r += (mean_br - self.auto_offset_brow_r) * t
                            self.auto_offset_inner_l += (mean_il - self.auto_offset_inner_l) * t
                            self.auto_offset_inner_r += (mean_ir - self.auto_offset_inner_r) * t
                            self.auto_offset_outer_l += (mean_ol - self.auto_offset_outer_l) * t
                            self.auto_offset_outer_r += (mean_or - self.auto_offset_outer_r) * t

                            self.lbl_auto_status.setText("Locked on neutral")
                            self.lbl_auto_status.setStyleSheet("color: #4CAF50;")
                        else:
                            self.lbl_auto_status.setText("Watching for drift...")
                            self.lbl_auto_status.setStyleSheet("color: #888;")

                    # Apply corrective offset to ALL channels
                    raw_brow_l -= self.auto_offset_brow_l
                    raw_brow_r -= self.auto_offset_brow_r
                    raw_inner_l -= self.auto_offset_inner_l
                    raw_inner_r -= self.auto_offset_inner_r
                    raw_outer_l -= self.auto_offset_outer_l
                    raw_outer_r -= self.auto_offset_outer_r
                    
                    self.last_raw_brow_l = raw_brow_l
                    self.last_raw_brow_r = raw_brow_r

                    # Apply symmetry calibration (offset + scale)
                    raw_brow_l = (raw_brow_l - self.sym_offset_l) * self.sym_scale_l
                    raw_brow_r = (raw_brow_r - self.sym_offset_r) * self.sym_scale_r
                    raw_inner_l = (raw_inner_l - self.sym_offset_l) * self.sym_scale_l
                    raw_inner_r = (raw_inner_r - self.sym_offset_r) * self.sym_scale_r
                    raw_outer_l = (raw_outer_l - self.sym_offset_l) * self.sym_scale_l
                    raw_outer_r = (raw_outer_r - self.sym_offset_r) * self.sym_scale_r

                    # Feed real inference results to predictive interpolators
                    smooth = max(0.01, 1.0 - (self.slider_smooth.value() / 100.0))
                    self.ema_left.smooth = smooth
                    self.ema_right.smooth = smooth
                    self.ema_inner_left.smooth = smooth
                    self.ema_inner_right.smooth = smooth
                    self.ema_outer_left.smooth = smooth
                    self.ema_outer_right.smooth = smooth

                    self.ema_left.update(raw_brow_l)
                    self.ema_right.update(raw_brow_r)
                    self.ema_inner_left.update(raw_inner_l)
                    self.ema_inner_right.update(raw_inner_r)
                    self.ema_outer_left.update(raw_outer_l)
                    self.ema_outer_right.update(raw_outer_r)

                    # Extrapolate at 100 Hz (velocity-predicted)
                    out_l = max(-1.0, min(1.0, (self.ema_left.value or 0.0) - self.offset_l))
                    out_r = max(-1.0, min(1.0, (self.ema_right.value or 0.0) - self.offset_r))
                    inner_l = max(-1.0, min(1.0, self.ema_inner_left.value or 0.0))
                    inner_r = max(-1.0, min(1.0, self.ema_inner_right.value or 0.0))
                    outer_l = max(-1.0, min(1.0, self.ema_outer_left.value or 0.0))
                    outer_r = max(-1.0, min(1.0, self.ema_outer_right.value or 0.0))

                    lbl_suffix = ""
                except Exception as e:
                    self._show_error_dialog("Inference Error", f"Inference failed:\n{e}", key="inference")
                    out_l = 0.0
                    out_r = 0.0
                    inner_l = 0.0
                    inner_r = 0.0
                    outer_l = 0.0
                    outer_r = 0.0
                    lbl_suffix = " (ERROR)"
            else:
                out_l = 0.0
                out_r = 0.0
                inner_l = 0.0
                inner_r = 0.0
                outer_l = 0.0
                outer_r = 0.0
                lbl_suffix = ""
                
            # Apply Synchronization Blending
            sync_factor = self.slider_sync.value() / 100.0
            if sync_factor > 0.0:
                avg_val = (out_l + out_r) / 2.0
                out_l = out_l * (1.0 - sync_factor) + (avg_val * sync_factor)
                out_r = out_r * (1.0 - sync_factor) + (avg_val * sync_factor)
            
            # Apply deadzone + boost (per-parameter)
            out_l = self._apply_deadzone_boost_param(out_l, "BrowExpressionLeft")
            out_r = self._apply_deadzone_boost_param(out_r, "BrowExpressionRight")
            inner_l = self._apply_deadzone_boost_param(inner_l, "BrowInnerUpLeft")
            inner_r = self._apply_deadzone_boost_param(inner_r, "BrowInnerUpRight")
            outer_l = self._apply_deadzone_boost_param(outer_l, "BrowOuterUpLeft")
            outer_r = self._apply_deadzone_boost_param(outer_r, "BrowOuterUpRight")

            if self.tabs.currentIndex() == 0:
                self.graph_history_l.append(out_l)
                self.graph_history_r.append(out_r)
                if len(self.graph_history_l) > 120: self.graph_history_l.pop(0)
                if len(self.graph_history_r) > 120: self.graph_history_r.pop(0)
                self.graph_widget.set_data(self.graph_history_l, self.graph_history_r)
                
            if self.tabs.currentIndex() == 0:
                self.lbl_l_brow.setText(f"Brow Slider: {out_l:.2f}{lbl_suffix}")
                self.lbl_r_brow.setText(f"Brow Slider: {out_r:.2f}{lbl_suffix}")

            osc_values = self._compute_osc_values(out_l, out_r, inner_l, inner_r, outer_l, outer_r)
            self.osc_param_values.update(osc_values)

            # Broadcast OSC if enabled
            if self.osc_enabled and self.osc_client:
                try:
                    for key, val in osc_values.items():
                        if self.osc_param_enabled.get(key, True):
                            self.osc_client.send_message(f"/avatar/parameters/FT/v2/{key}", float(val))
                except Exception as e:
                    self._show_error_dialog("OSC Error", f"Failed to send OSC:\n{e}", key="osc_send")
                    self.osc_enabled = False
                    self.osc_client = None
                    for btn in (self.btn_osc, self.btn_osc_main):
                        btn.setText("Start OSC Sender")
                        btn.setProperty("class", "success-btn")
                        self._refresh_button_style(btn)
                    self.txt_ip.setEnabled(True)
                    self.txt_port.setEnabled(True)
            self._tick_osc_fps()

    def _tick_osc_fps(self):
        self._osc_frame_count += 1
        now = time.time()
        elapsed = now - self._osc_fps_time
        if elapsed >= 0.5:
            self._osc_fps = self._osc_frame_count / elapsed
            self._osc_frame_count = 0
            self._osc_fps_time = now

    def _start_training_setup(self):
        reply = QMessageBox.question(
            self, "Setup Training Environment",
            "This will download Python 3.10 + PyTorch CPU (~300MB).\n"
            "This is a one-time setup for model baking.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self.btn_setup_training.setEnabled(False)
        self.btn_setup_training.setText("Setting up...")
        self._setup_thread = TrainingSetupThread()
        self._setup_thread.progress.connect(lambda msg: print(f"[Setup] {msg}"))
        self._setup_thread.progress.connect(self._update_train_status_smart)
        self._setup_thread.finished.connect(self._training_setup_finished)
        self._setup_thread.start()

    def _training_setup_finished(self, success):
        if success:
            self.btn_setup_training.setText("Training Environment Ready")
            self.lbl_train_status.setText("Status: Setup complete. You can now bake models.")
        else:
            self.btn_setup_training.setText("Setup Training Environment")
            self.btn_setup_training.setEnabled(True)
            self.lbl_train_status.setText("Status: Setup failed. Check console for details.")

    def start_training(self):
        if len(self.recorded_frames) < 10:
            QMessageBox.warning(self, "Warning", "Not enough data! Please record frames first.")
            return

        self._start_training(self.eyebrow_images_dir, self.csv_path, self.val_csv_path)

    def start_training_with_path(self):
        base_dir = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", str(self.data_dir))
        if not base_dir:
            return
        
        base_dir = Path(base_dir)
        data_dir = base_dir / "eyebrow_images"
        train_csv = base_dir / "train.csv"
        val_csv = base_dir / "val.csv"
        
        missing = []
        if not data_dir.exists():
            missing.append(str(data_dir))
        if not train_csv.exists():
            missing.append(str(train_csv))
        if not val_csv.exists():
            missing.append(str(val_csv))
        
        if missing:
            msg = "Missing required dataset files/folder:\n" + "\n".join(missing)
            QMessageBox.warning(self, "Warning", msg)
            return
        
        self._start_training(data_dir, train_csv, val_csv)

    def _start_training(self, data_dir, train_csv, val_csv):
        self._set_training_buttons(False)
        model_dir = self._get_model_dir()
        self.thread = TrainingThread(data_dir, train_csv, val_csv, model_dir, parent=self)
        self.thread.progress.connect(self._update_train_status_smart)
        self.thread.progress.connect(self._log_train_important)
        self.thread.finished.connect(self.training_finished)
        self.thread.start()
        
    def _set_training_buttons(self, enabled):
        self.btn_train.setEnabled(enabled)
        self.btn_train_with_path.setEnabled(enabled)

    def _refresh_button_style(self, btn):
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def _get_model_dir(self):
        appdata = os.getenv("APPDATA")
        if appdata:
            base_dir = Path(appdata) / "VREyebrowTracker"
        else:
            base_dir = Path("data")
        model_dir = base_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        return str(model_dir)

    @staticmethod
    def _power_curve(x, gamma, boost_pos, boost_neg):
        """Smooth power-curve remapping. No hard deadzone — just compression near zero.

        y = sign(x) * |x|^gamma * boost

        gamma=1.0: linear (no compression)
        gamma=2.0: quadratic (small values strongly dampened)
        gamma=3.0: cubic (very strong dampening near zero)

        At |x|=0.1, gamma=2: output=0.01 (barely visible, won't trigger angry)
        At |x|=0.5, gamma=2: output=0.25 (moderate expression)
        At |x|=1.0, gamma=any: output=1.0 (full range always preserved)
        """
        if x == 0.0:
            return 0.0
        sign = 1.0 if x >= 0 else -1.0
        boost = boost_pos if sign >= 0 else boost_neg
        y = abs(x) ** gamma * boost
        return max(-1.0, min(1.0, sign * y))

    def _apply_deadzone_boost_param(self, x, key):
        if key in getattr(self, "param_deadzone_sliders", {}):
            curve = self.param_deadzone_sliders[key].value() / 100.0
            boost_pos = self.param_boost_pos_sliders[key].value() / 100.0
            boost_neg = self.param_boost_neg_sliders[key].value() / 100.0
        else:
            curve = 0.05  # mild default
            boost_pos = 1.0
            boost_neg = 1.0
        # Map curve slider (0-0.30) → gamma (1.0 to 4.0)
        gamma = 1.0 + curve * 10.0
        return self._power_curve(x, gamma, boost_pos, boost_neg)

    def _toggle_param_osc(self, key, state):
        self.osc_param_enabled[key] = (state == Qt.Checked)
        self._update_setting(f"osc_enable_{key}", self.osc_param_enabled[key])

    def _toggle_osc_debug(self, state):
        self.osc_debug_enabled = (state == Qt.Checked)
        if hasattr(self, "debug_panel"):
            self.debug_panel.setVisible(self.osc_debug_enabled)


    def _compute_osc_values(self, brow_l, brow_r, inner_l, inner_r, outer_l, outer_r):
        up_l = max(0.0, brow_l)
        up_r = max(0.0, brow_r)
        down_l = max(0.0, -brow_l)
        down_r = max(0.0, -brow_r)
        avg = (brow_l + brow_r) / 2.0
        return {
            "BrowExpressionLeft": float(brow_l),
            "BrowExpressionRight": float(brow_r),
            "BrowUpLeft": float(up_l),
            "BrowUpRight": float(up_r),
            "BrowDownLeft": float(down_l),
            "BrowDownRight": float(down_r),
            "BrowUp": float(max(0.0, avg)),
            "BrowDown": float(max(0.0, -avg)),
        }

    def _show_error_dialog(self, title, message, key=None, interval=3.0):
        if key is None:
            QMessageBox.warning(self, title, message)
            return
        now = time.time()
        last_time = self._err_dialog_last.get(key, 0.0)
        if now - last_time >= interval:
            QMessageBox.warning(self, title, message)
            self._err_dialog_last[key] = now
        
    def update_training_status(self, msg):
        self.lbl_train_status.setText(f"Status: {msg}")

    def _update_train_status_smart(self, msg):
        """Show only key status updates in the label."""
        ml = msg.lower()
        if any(kw in ml for kw in ['error', 'complete', 'done', 'saved', 'exported',
                                     'searching', 'using:', 'training', 'epoch']):
            if 'epoch' in ml:
                self.lbl_train_status.setText(f"Status: {msg.strip()[:80]}")
            else:
                self.lbl_train_status.setText(f"Status: {msg.strip()}")

    def _log_train_important(self, msg):
        """Log to console. Progress bars overwrite the last line instead of scrolling."""
        line = msg.strip()
        if not line:
            return
        if any(x in line for x in ['%|', 'it/s', 'it\\s']):
            # Progress bar: overwrite last line
            if hasattr(self, 'txt_console'):
                cursor = self.txt_console.textCursor()
                cursor.movePosition(cursor.End)
                cursor.movePosition(cursor.StartOfBlock, cursor.KeepAnchor)
                cursor.removeSelectedText()
                cursor.insertText(f"[Train] {line}")
                self.txt_console.setTextCursor(cursor)
        else:
            print(f"[Train] {line}")

    def training_finished(self, new_model_path):
        self._set_training_buttons(True)
        if new_model_path:
            try:
                if self.load_weights(new_model_path):
                    self._update_setting("last_model_path", self.current_model_path)
                    self.lbl_current_model.setText(f"{os.path.basename(self.current_model_path)} (Newly Trained!)")
            except Exception as e:
                print(f"Failed to load newly trained model: {e}")

    def _append_log(self, text):
        if not hasattr(self, "txt_console"):
            return
        if not text:
            return
        clean = self._ansi_re.sub("", text)
        parts = clean.split("\r")
        for i, part in enumerate(parts):
            if i == 0:
                self._append_console_text(part)
            else:
                self._replace_console_line(part)
        self.txt_console.moveCursor(QTextCursor.End)

    def _append_console_text(self, text):
        if not text:
            return
        cursor = self.txt_console.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.txt_console.setTextCursor(cursor)

    def _replace_console_line(self, text):
        cursor = self.txt_console.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertText(text)
        self.txt_console.setTextCursor(cursor)

    def closeEvent(self, event):
        # Stop camera threads first to release device handles
        if getattr(self, "cam_left", None):
            try:
                self.cam_left.stop(wait_ms=1000)
            except Exception:
                pass
            self.cam_left = None
        if getattr(self, "cam_right", None):
            try:
                self.cam_right.stop(wait_ms=1000)
            except Exception:
                pass
            self.cam_right = None
        if self.mjpeg_server.is_running:
            self.mjpeg_server.stop()
        if hasattr(self, "_stdout"):
            sys.stdout = self._stdout
        if hasattr(self, "_stderr"):
            sys.stderr = self._stderr
        self._save_settings()
        super().closeEvent(event)

if __name__ == '__main__':
    log_path = None
    try:
        appdata = os.getenv("APPDATA")
        if appdata:
            log_dir = Path(appdata) / "VREyebrowTracker"
        else:
            log_dir = Path("data")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "gui_startup.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"--- GUI start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        if hasattr(Qt, 'AA_EnableHighDpiScaling'):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        app = QApplication(sys.argv)
        ex = VREyebrowTrackerGUI()
        ex.show()
        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[FATAL] {e}\n{tb}\n")
        except Exception:
            pass
        print(f"[FATAL] GUI failed to start: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")
