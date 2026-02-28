import sys
import os
import platform
import winreg
import time
import re
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
import cv2
import torch
import torchvision
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
                             QScrollArea, QGridLayout, QComboBox, QTextEdit, QPlainTextEdit, QStackedLayout, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QPainter, QPen, QBrush, QTextCursor
from torchvision import transforms
# Proxy imports
import socket
import json
import struct
from pythonosc.udp_client import SimpleUDPClient

from model import TinyBrowNet
from inference import EMARegressor, setup_transform, crop_roi
from train import train_model

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
                            time.sleep(1)
                            if isinstance(self.source, int):
                                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                                if not self.cap.isOpened():
                                    self.cap = cv2.VideoCapture(self.source)
                            else:
                                try:
                                    self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                                except Exception:
                                    self.cap = cv2.VideoCapture(self.source)
                                if not self.cap.isOpened():
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
    result = pyqtSignal(list)
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
            self.result.emit(available)
        except Exception as e:
            self.error.emit(str(e))

class TrainingThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, data_dir, train_csv, val_csv, model_dir, use_lr_models=False, parent=None):
        super().__init__(parent)
        self.data_dir = str(data_dir)
        self.train_csv = str(train_csv)
        self.val_csv = str(val_csv)
        self.model_dir = str(model_dir)
        self.use_lr_models = use_lr_models
    
    def run(self):
        self.progress.emit("Starting PyTorch Training...")
        try:
            import importlib
            import train
            importlib.reload(train)
            from train import train_model, train_model_pair

            # start with values passed into constructor
            DATA_DIR = self.data_dir
            TRAIN_CSV = self.train_csv
            VAL_CSV = self.val_csv
            self.progress.emit("Training in progress (check terminal for logs)...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            model_dir = Path(self.model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            if self.use_lr_models:
                save_left = str(model_dir / f"model_left_{timestamp}.pth")
                save_right = str(model_dir / f"model_right_{timestamp}.pth")
                ok = train_model_pair(DATA_DIR, TRAIN_CSV, VAL_CSV, save_left=save_left, save_right=save_right)
                if ok:
                    self.progress.emit(f"Training Complete! Saved '{save_left}' and '{save_right}'.")
                    save_path = (save_left, save_right)
                else:
                    self.progress.emit("Error: Training failed. Check dataset paths and logs.")
                    save_path = ""
            else:
                save_path = str(model_dir / f"model_{timestamp}.pth")
                ok = train_model(DATA_DIR, TRAIN_CSV, VAL_CSV, save_path=save_path)
                if ok:
                    self.progress.emit(f"Training Complete! Saved '{save_path}'.")
                else:
                    self.progress.emit("Error: Training failed. Check dataset paths and logs.")
                    save_path = ""
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
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
        self.device = self.available_devices[0][1]
        self.model = TinyBrowNet().to(self.device)
        self.model_left = TinyBrowNet().to(self.device)
        self.model_right = TinyBrowNet().to(self.device)
        self.use_lr_models = False
        self.model_main_has_inner_outer = True
        self.model_left_has_inner_outer = True
        self.model_right_has_inner_outer = True
        self.model.eval()
        self.model_left.eval()
        self.model_right.eval()
        self.current_model_path = "None Loaded"
        self.current_model_left_path = ""
        self.current_model_right_path = ""
            
        self.transform = setup_transform()
        self.ema_left = EMARegressor(alpha=0.3)
        self.ema_right = EMARegressor(alpha=0.3)
        self.ema_inner_left = EMARegressor(alpha=0.3)
        self.ema_inner_right = EMARegressor(alpha=0.3)
        self.ema_outer_left = EMARegressor(alpha=0.3)
        self.ema_outer_right = EMARegressor(alpha=0.3)
        
        self.offset_l = 0.0
        self.offset_r = 0.0
        
        self.current_fps = 0.0
        self.last_update_time = time.time()
        
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
        
        # Auto-Baseline State
        self.brow_history_l = []
        self.brow_history_r = []
        self.graph_history_l = []
        self.graph_history_r = []
        self.ref_pts_l = None
        self.ref_pts_r = None
        self.auto_offset_l = 0.0
        self.auto_offset_r = 0.0
        self._err_dialog_last = {}
        self._ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
        self.osc_debug_enabled = False
        self.osc_param_order = [
            "BrowExpressionLeft",
            "BrowExpressionRight",
            "BrowInnerUpLeft",
            "BrowInnerUpRight",
            "BrowOuterUpLeft",
            "BrowOuterUpRight",
        ]
        self.osc_param_values = {k: 0.0 for k in self.osc_param_order}
        self.osc_param_labels = {
            "BrowExpressionLeft": "BrowL",
            "BrowExpressionRight": "BrowR",
            "BrowInnerUpLeft": "InnerL",
            "BrowInnerUpRight": "InnerR",
            "BrowOuterUpLeft": "OuterL",
            "BrowOuterUpRight": "OuterR",
        }
        self.osc_param_enabled = {k: True for k in self.osc_param_order}
        self.use_combined_feed = False
        self.combined_rotate = 0
        self.hmd_profile = "DIY"
        
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
        
        self.init_ui()
        self.apply_theme()
        self.update_dataset_status()
        self.apply_settings()
        QTimer.singleShot(200, self.scan_cameras)

    def _get_available_devices(self):
        devices = []
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                devices.append((f"GPU {i}: {name}", torch.device(f"cuda:{i}")))
        cpu_name = ""
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                cpu_name = cpu_name.strip()
        except Exception:
            cpu_name = platform.processor().strip()
        if cpu_name:
            devices.append((f"CPU: {cpu_name}", torch.device("cpu")))
        else:
            devices.append(("CPU", torch.device("cpu")))
        return devices

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
        self.use_combined_feed = is_bigscreen
        self._update_setting("combined_feed", self.use_combined_feed)
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
        # Auto-select camera if only one device (Bigeye-like)
        if is_bigscreen and hasattr(self, "cmb_cam_l") and self.camera_devices:
            if self.cmb_cam_l.currentData() == "url":
                # Try to auto-select Bigeye by friendly name if available
                if self.camera_friendly_names:
                    for idx, name in enumerate(self.camera_friendly_names[:len(self.camera_devices)]):
                        if "bigeye" in name.lower():
                            self._set_camera_combo(self.cmb_cam_l, self.camera_devices[idx])
                            return
                # Fallback: if only one device, select it
                if len(self.camera_devices) == 1:
                    self._set_camera_combo(self.cmb_cam_l, self.camera_devices[0])

    def _toggle_combined_feed(self, state):
        self.use_combined_feed = (state == Qt.Checked)

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
        mid = w // 2
        left = frame_bgr[:, :mid]
        right = frame_bgr[:, mid:]
        return left, right

    def _get_camera_source(self, combo, url_text):
        data = combo.currentData()
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

    def _apply_camera_scan(self, devices):
        self.camera_devices = list(devices)
        self.camera_friendly_names = self._get_camera_friendly_names()
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
        if "deadzone" in s and hasattr(self, "slider_deadzone"):
            self.slider_deadzone.setValue(int(s["deadzone"]))
        if "boost" in s and hasattr(self, "slider_boost_pos") and hasattr(self, "slider_boost_neg"):
            self.slider_boost_pos.setValue(int(s["boost"]))
            self.slider_boost_neg.setValue(int(s["boost"]))
        if "boost_pos" in s and hasattr(self, "slider_boost_pos"):
            self.slider_boost_pos.setValue(int(s["boost_pos"]))
        if "boost_neg" in s and hasattr(self, "slider_boost_neg"):
            self.slider_boost_neg.setValue(int(s["boost_neg"]))
        for k in self.osc_param_order:
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
            if enable_key in s and k in self.osc_param_toggles:
                self.osc_param_toggles[k].setChecked(bool(s[enable_key]))
        if "auto_baseline" in s:
            self.chk_auto_baseline.setChecked(bool(s["auto_baseline"]))
        if "alpha" in s:
            self.slider_alpha.setValue(int(s["alpha"]))
        if "device_index" in s and isinstance(s["device_index"], int):
            idx = s["device_index"]
            if 0 <= idx < self.cmb_device.count():
                self.cmb_device.setCurrentIndex(idx)
        if "use_lr_models" in s:
            self.use_lr_models = bool(s["use_lr_models"])
            if hasattr(self, "chk_lr_models"):
                self.chk_lr_models.setChecked(self.use_lr_models)
        if "hmd_profile" in s and hasattr(self, "cmb_hmd"):
            self._set_hmd_combo(s["hmd_profile"])
        if "combined_feed" in s:
            self.use_combined_feed = bool(s["combined_feed"])
            if hasattr(self, "chk_combined"):
                self.chk_combined.setChecked(self.use_combined_feed)
        if "combined_rotate" in s and hasattr(self, "cmb_combined_rotate"):
            self._set_camera_combo(self.cmb_combined_rotate, s["combined_rotate"])
        if "last_model_path" in s:
            path = s["last_model_path"]
            if path and os.path.exists(path):
                self.load_weights(path)
                self.lbl_current_model.setText(os.path.basename(path))
        if "last_model_left_path" in s and "last_model_right_path" in s:
            lpath = s.get("last_model_left_path")
            rpath = s.get("last_model_right_path")
            if lpath and rpath and os.path.exists(lpath) and os.path.exists(rpath):
                self.load_weights_lr(lpath, rpath)
        if hasattr(self, "cmb_hmd"):
            self._apply_hmd_ui()

    def on_device_changed(self, idx):
        if idx < 0 or idx >= len(self.available_devices):
            return
        label, device = self.available_devices[idx]
        if device == self.device:
            return
        self.device = device
        self._update_setting("device_index", idx)
        try:
            self.model.to(self.device)
            self.model_left.to(self.device)
            self.model_right.to(self.device)
        except Exception as e:
            QMessageBox.warning(self, "Device Error", f"Failed to switch to {label}:\n{e}")
            self.device = torch.device("cpu")
            self.model.to(self.device)
            self.model_left.to(self.device)
            self.model_right.to(self.device)
            cpu_idx = next((i for i, (lbl, _d) in enumerate(self.available_devices) if lbl.startswith("CPU")), None)
            if cpu_idx is not None:
                self.cmb_device.setCurrentIndex(cpu_idx)
        
    def _load_state_dict_compat(self, model, state_dict):
        current_dict = model.state_dict()
        has_inner_outer = True
        if 'fc2.weight' in state_dict and 'fc2.weight' in current_dict:
            out_old = state_dict['fc2.weight'].shape[0]
            out_new = current_dict['fc2.weight'].shape[0]
            if out_new >= 3 and out_old < 3:
                has_inner_outer = False
            if out_old != out_new:
                new_fc2_weight = current_dict['fc2.weight'].clone()
                new_fc2_bias = current_dict['fc2.bias'].clone()
                copy_n = min(out_old, out_new)
                new_fc2_weight[:copy_n, :] = state_dict['fc2.weight'][:copy_n, :]
                new_fc2_bias[:copy_n] = state_dict['fc2.bias'][:copy_n]
                state_dict['fc2.weight'] = new_fc2_weight
                state_dict['fc2.bias'] = new_fc2_bias
        model.load_state_dict(state_dict)
        model.eval()
        return has_inner_outer

    def load_weights(self, path):
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model_main_has_inner_outer = self._load_state_dict_compat(self.model, state_dict)
            self.current_model_path = path
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False

    def load_weights_lr(self, left_path, right_path):
        try:
            state_left = torch.load(left_path, map_location=self.device)
            state_right = torch.load(right_path, map_location=self.device)
            self.model_left_has_inner_outer = self._load_state_dict_compat(self.model_left, state_left)
            self.model_right_has_inner_outer = self._load_state_dict_compat(self.model_right, state_right)
            self.current_model_left_path = left_path
            self.current_model_right_path = right_path
            self._update_setting("last_model_left_path", left_path)
            self._update_setting("last_model_right_path", right_path)
            return True
        except Exception as e:
            print(f"Error loading L/R weights: {e}")
            return False

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top Bar (Connection)
        top_bar = QHBoxLayout()
        self.btn_connect_left = QPushButton("Start Left Stream")
        self.btn_connect_left.setProperty("class", "primary-btn")
        self.btn_connect_left.clicked.connect(self.toggle_left_connection)
        top_bar.addWidget(self.btn_connect_left)
        
        self.btn_connect_right = QPushButton("Start Right Stream")
        self.btn_connect_right.setProperty("class", "primary-btn")
        self.btn_connect_right.clicked.connect(self.toggle_right_connection)
        top_bar.addWidget(self.btn_connect_right)

        theme_hmd_col = QVBoxLayout()
        self.btn_theme = QPushButton("Light Mode")
        self.btn_theme.setProperty("class", "theme-btn")
        self.btn_theme.clicked.connect(self.toggle_theme)
        theme_hmd_col.addWidget(self.btn_theme)

        self.cmb_hmd = QComboBox()
        self.cmb_hmd.addItem("Pimax Crystal / Super QLED")
        self.cmb_hmd.addItem("VIVE PRO EYE")
        self.cmb_hmd.addItem("Varjo")
        self.cmb_hmd.addItem("HP Reverb G2")
        self.cmb_hmd.addItem("Pimax Dream Air")
        self.cmb_hmd.addItem("Pimax Crystal Super uOLED")
        self.cmb_hmd.addItem("Bigscreen Beyond 2e")
        self.cmb_hmd.addItem("DIY")
        self.cmb_hmd.currentIndexChanged.connect(self._on_hmd_changed)
        theme_hmd_col.addWidget(self.cmb_hmd)

        top_bar.addLayout(theme_hmd_col)

        
        main_layout.addLayout(top_bar)
        
        # TABS
        self.tabs = QTabWidget()
        self.tab_tracker = QWidget()
        self.tab_calibration = QWidget()
        self.tab_console = QWidget()
        self.tabs.addTab(self.tab_tracker, "1. Live Tracker & OSC")
        self.tabs.addTab(self.tab_calibration, "2. Calibration & Training")
        self.tabs.addTab(self.tab_console, "3. Console")
        main_layout.addWidget(self.tabs)
        
        self.setup_tracker_tab()
        self.setup_calibration_tab()
        self.setup_console_tab()
        
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
        
        self.txt_cam_l = QLineEdit("http://127.0.0.1:5555/eye/left")
        self.txt_cam_l.setPlaceholderText("Left Camera ESP32 URL...")
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
        self.left_eye_box.addWidget(lbl_l)
        self.left_eye_box.addWidget(self.lbl_l_brow)
        
        # Right Eye
        self.right_eye_box = QVBoxLayout()
        
        self.lbl_r_fps = QLabel("FPS: 0")
        self.lbl_r_fps.setAlignment(Qt.AlignLeft)
        self.lbl_r_fps.setProperty("class", "muted-label")
        
        self.txt_cam_r = QLineEdit("http://127.0.0.1:5555/eye/right")
        self.txt_cam_r.setPlaceholderText("Right Camera ESP32 URL...")
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

        # OSC Debug Widget (debug page)
        self.osc_debug_widget = ParamBarGraphWidget()
        self.osc_debug_widget.set_data([(self.osc_param_labels.get(k, k), 0.0) for k in self.osc_param_order])
        self.values_row_widget = ParamBarGraphWidget(show_labels=False, show_values=True, show_bars=False)
        self.values_row_widget.setMinimumHeight(24)
        self.values_row_widget.set_data([(self.osc_param_labels.get(k, k), 0.0) for k in self.osc_param_order])

        # OSC enable toggles (per-parameter)
        self.osc_param_toggles = {}
        toggles_row = QWidget()
        toggles_layout = QHBoxLayout(toggles_row)
        toggles_layout.setContentsMargins(0, 0, 0, 0)
        for k in self.osc_param_order:
            chk = QCheckBox("")
            chk.setChecked(True)
            chk.setToolTip(f"OSC Enable: {self.osc_param_labels.get(k, k)}")
            chk.stateChanged.connect(lambda state, key=k: self._toggle_param_osc(key, state))
            chk.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.osc_param_toggles[k] = chk
            toggles_layout.addWidget(chk)

        # Per-Parameter Tuning (Debug)
        self.param_deadzone_sliders = {}
        self.param_boost_pos_sliders = {}
        self.param_boost_neg_sliders = {}
        self.param_group = QGroupBox("Per-Parameter Tuning (Debug)")
        param_layout = QGridLayout()
        param_layout.addWidget(QLabel("Param"), 0, 0)
        param_layout.addWidget(QLabel("Deadzone"), 0, 1)
        param_layout.addWidget(QLabel("Boost +"), 0, 2)
        param_layout.addWidget(QLabel("Boost -"), 0, 3)
        row = 1
        for k in self.osc_param_order:
            label = QLabel(self.osc_param_labels.get(k, k))
            dz = QSlider(Qt.Horizontal)
            dz.setRange(0, 30)
            dz.setValue(5)
            dz.valueChanged.connect(lambda v, key=k: self._update_setting(f"deadzone_{key}", v))
            boost_pos = QSlider(Qt.Horizontal)
            boost_pos.setRange(50, 300)
            boost_pos.setValue(100)
            boost_pos.valueChanged.connect(lambda v, key=k: self._update_setting(f"boost_pos_{key}", v))
            boost_neg = QSlider(Qt.Horizontal)
            boost_neg.setRange(50, 300)
            boost_neg.setValue(100)
            boost_neg.valueChanged.connect(lambda v, key=k: self._update_setting(f"boost_neg_{key}", v))
            self.param_deadzone_sliders[k] = dz
            self.param_boost_pos_sliders[k] = boost_pos
            self.param_boost_neg_sliders[k] = boost_neg
            param_layout.addWidget(label, row, 0)
            param_layout.addWidget(dz, row, 1)
            param_layout.addWidget(boost_pos, row, 2)
            param_layout.addWidget(boost_neg, row, 3)
            row += 1
        self.param_group.setLayout(param_layout)

        debug_page = QWidget()
        debug_page_layout = QVBoxLayout(debug_page)
        debug_page_layout.addWidget(self.osc_debug_widget)
        debug_page_layout.addWidget(self.values_row_widget)
        debug_page_layout.addWidget(toggles_row)
        debug_page_layout.addWidget(self.param_group)
        # Manual Override (debug only)
        debug_page_layout.addWidget(grp_manual)
        debug_page_layout.addStretch(1)

        self.graph_stack = QStackedLayout()
        self.graph_stack.addWidget(self.grp_graph)
        self.graph_stack.addWidget(debug_page)
        self.graph_stack_widget = QWidget()
        self.graph_stack_widget.setLayout(self.graph_stack)
        self.graph_stack_widget.setMaximumHeight(220)
        eye_layout.addWidget(self.graph_stack_widget)

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
        # Right Side (Settings)
        # ====================
        settings_panel = QVBoxLayout()

        # Device Selection Group
        grp_device = QGroupBox("Compute Device")
        grp_device_layout = QVBoxLayout()
        self.cmb_device = QComboBox()
        for label, _dev in self.available_devices:
            self.cmb_device.addItem(label)
        self.cmb_device.currentIndexChanged.connect(self.on_device_changed)
        grp_device_layout.addWidget(self.cmb_device)
        grp_device.setLayout(grp_device_layout)
        settings_panel.addWidget(grp_device)
        
        # Model Selection Group
        grp_model = QGroupBox("Model Configuration")
        grp_model_layout = QVBoxLayout()
        
        btn_load_model = QPushButton("Load Eyebrow Weights (.pth)")
        btn_load_model.clicked.connect(self.browse_weights)
        grp_model_layout.addWidget(btn_load_model)
        
        self.lbl_current_model = QLabel("Eyebrow: None Loaded")
        self.lbl_current_model.setProperty("class", "muted-label")
        grp_model_layout.addWidget(self.lbl_current_model)
        
        grp_model.setLayout(grp_model_layout)
        settings_panel.addWidget(grp_model)
        
        # OSC Group
        grp_osc = QGroupBox("OSC")
        grp_osc_layout = QVBoxLayout()
        
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("AI Output IP:"))
        self.txt_ip = QLineEdit("127.0.0.1")
        self.txt_ip.textChanged.connect(lambda v: self._update_setting("osc_ip", v))
        port_layout.addWidget(self.txt_ip)
        port_layout.addWidget(QLabel("Port (VRChat):"))
        self.txt_port = QLineEdit("9000")
        self.txt_port.setFixedWidth(50)
        self.txt_port.textChanged.connect(lambda v: self._update_setting("osc_port", v))
        port_layout.addWidget(self.txt_port)
        grp_osc_layout.addLayout(port_layout)
        
        # Proxy Layout Removed
        self.btn_osc = QPushButton("Start OSC Sender")
        self.btn_osc.setProperty("class", "success-btn")
        self.btn_osc.clicked.connect(self.toggle_osc)
        grp_osc_layout.addWidget(self.btn_osc)
        
        grp_osc.setLayout(grp_osc_layout)
        settings_panel.addWidget(grp_osc)
        
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
        grp_sync = QGroupBox("L/R Synchronization")
        grp_sync_layout =QVBoxLayout()
        self.lbl_sync_val = QLabel("Sync: 0%")
        grp_sync_layout.addWidget(self.lbl_sync_val)
        
        self.slider_sync = QSlider(Qt.Horizontal)
        self.slider_sync.setRange(0, 100)
        self.slider_sync.setValue(0)
        self.slider_sync.valueChanged.connect(lambda v: self.lbl_sync_val.setText(f"Sync: {v}%"))
        self.slider_sync.valueChanged.connect(lambda v: self._update_setting("sync", v))
        grp_sync_layout.addWidget(self.slider_sync)
        
        grp_sync.setLayout(grp_sync_layout)
        settings_panel.addWidget(grp_sync)
        
        # --- Auto Baseline Correction panel ---
        auto_group = QGroupBox("Automatic HMD Position Compensation")
        auto_layout = QVBoxLayout()
        
        auto_controls = QHBoxLayout()
        self.chk_auto_baseline = QCheckBox("Enable Auto-Baseline")
        self.chk_auto_baseline.setChecked(False)
        self.chk_auto_baseline.stateChanged.connect(self.toggle_auto_baseline)
        self.chk_auto_baseline.stateChanged.connect(lambda v: self._update_setting("auto_baseline", v == Qt.Checked))
        
        self.btn_reset_baseline = QPushButton("Snapshot Baseline Now")
        self.btn_reset_baseline.clicked.connect(self.reset_auto_baseline)
        auto_controls.addWidget(self.chk_auto_baseline)
        auto_controls.addWidget(self.btn_reset_baseline)
        
        auto_sliders = QHBoxLayout()
        auto_sliders.addWidget(QLabel("Catch-up Speed (\u03B1):"))
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(1, 100)
        self.slider_alpha.setValue(5) # Default 5% alpha
        self.slider_alpha.valueChanged.connect(lambda v: self._update_setting("alpha", v))
        auto_sliders.addWidget(self.slider_alpha)

        self.btn_set_neutral = QPushButton("Set Neutral Baseline (Offset)")
        self.btn_set_neutral.setToolTip("Click while neutral to fix tracking if the headset shifted.")
        self.btn_set_neutral.clicked.connect(self.set_neutral_baseline)
        self.btn_reset_offsets = QPushButton("Reset Offsets (Zero)")
        self.btn_reset_offsets.setToolTip("Force offsets to 0.0 for both eyes.")
        self.btn_reset_offsets.clicked.connect(self.reset_offsets_zero)
        
        self.lbl_auto_status = QLabel("Status: Waiting for stable neutral frame...")
        self.lbl_auto_status.setStyleSheet("color: #888;")
        
        auto_layout.addLayout(auto_controls)
        auto_layout.addLayout(auto_sliders)
        auto_layout.addWidget(self.btn_set_neutral)
        auto_layout.addWidget(self.btn_reset_offsets)
        auto_layout.addWidget(self.lbl_auto_status)
        auto_group.setLayout(auto_layout)
        settings_panel.addWidget(auto_group)
        # -------------------------------------
        
        settings_panel.addStretch()
        layout.addLayout(settings_panel, stretch=1)
        
    def toggle_auto_baseline(self, state):
        if state == 0: # Unchecked
            self.lbl_auto_status.setText("Status: Disabled")
            self.lbl_auto_status.setStyleSheet("color: #888;")
        else:
            self.lbl_auto_status.setText("Status: Waiting for stable neutral frame...")
            self.lbl_auto_status.setStyleSheet("color: #888;")
            
    def reset_auto_baseline(self):
        self.ref_pts_l = None
        self.ref_pts_r = None
        self.auto_offset_l = 0.0
        self.auto_offset_r = 0.0
        self.brow_history_l.clear()
        self.brow_history_r.clear()
        self.lbl_auto_status.setText("Status: Baseline Reset. Waiting for face data...")
        self.lbl_auto_status.setStyleSheet("color: #888;")

    def setup_calibration_tab(self):
        layout = QVBoxLayout(self.tab_calibration)
        
        instr = QLabel("The manual buttons have been removed.\n"
                       "Click START, and follow the instructions on screen.\n"
                       "The system will automatically record your face for each expression.")
        instr.setProperty("class", "bold-label")
        layout.addWidget(instr)
        
        # Automatic Guided Sequence UI
        seq_controls = QHBoxLayout()
        
        self.btn_start_seq = QPushButton("START GUIDED CALIBRATION")
        self.btn_start_seq.setProperty("class", "primary-btn-success")
        self.btn_start_seq.clicked.connect(self.start_calibration_sequence)
        seq_controls.addWidget(self.btn_start_seq)
        
        self.btn_stop_seq = QPushButton("STOP CALIBRATION")
        self.btn_stop_seq.setProperty("class", "primary-btn-danger")
        self.btn_stop_seq.clicked.connect(self.stop_calibration_sequence)
        self.btn_stop_seq.setEnabled(False)
        seq_controls.addWidget(self.btn_stop_seq)
        
        seq_layout = QVBoxLayout()
        seq_layout.addLayout(seq_controls)
        
        self.lbl_seq_instruction = QLabel("Ready")
        self.lbl_seq_instruction.setAlignment(Qt.AlignCenter)
        self.lbl_seq_instruction.setFont(QFont("Arial", 24, QFont.Bold))
        self.lbl_seq_instruction.setStyleSheet("color: #111; margin-top: 20px; margin-bottom: 20px;")
        seq_layout.addWidget(self.lbl_seq_instruction)
        
        layout.addLayout(seq_layout)
        
        # Dataset Management
        data_layout = QHBoxLayout()
        self.lbl_dataset_status = QLabel("  |  Current Saved Dataset: 0 images")
        self.lbl_dataset_status.setProperty("class", "muted-label")
        data_layout.addWidget(self.lbl_dataset_status)
        
        self.btn_clear_data = QPushButton("Clear Calibration Data")
        self.btn_clear_data.setProperty("class", "primary-btn-danger")
        self.btn_clear_data.clicked.connect(self.clear_calibration_data)
        data_layout.addWidget(self.btn_clear_data)
        layout.addLayout(data_layout)
        
        # Training Section
        lbl_train = QLabel("Training Pipeline")
        lbl_train.setFont(QFont("Arial", 12, QFont.Bold))
        lbl_train.setProperty("class", "header-label")
        layout.addWidget(lbl_train)

        train_mode_row = QHBoxLayout()
        self.chk_lr_models = QCheckBox("Beta: Separate L/R Models")
        self.chk_lr_models.setChecked(False)
        self.chk_lr_models.stateChanged.connect(self._toggle_lr_models)
        self.chk_lr_models.stateChanged.connect(lambda v: self._update_setting("use_lr_models", v == Qt.Checked))
        train_mode_row.addWidget(self.chk_lr_models)
        train_mode_row.addStretch(1)
        layout.addLayout(train_mode_row)
        
        train_btn_row = QHBoxLayout()
        self.btn_train = QPushButton("BAKE MODEL (Start Training)")
        self.btn_train.setObjectName("btn_bake_main")
        self.btn_train.setProperty("class", "primary-btn-purple")
        self.btn_train.clicked.connect(self.start_training)
        train_btn_row.addWidget(self.btn_train)
        
        self.btn_train_with_path = QPushButton("BAKE (With Path)")
        self.btn_train_with_path.setObjectName("btn_bake_with_path")
        self.btn_train_with_path.setProperty("class", "primary-btn")
        self.btn_train_with_path.clicked.connect(self.start_training_with_path)
        train_btn_row.addWidget(self.btn_train_with_path)
        
        layout.addLayout(train_btn_row)
        
        self.lbl_train_status = QLabel("Status: Idle")
        layout.addWidget(self.lbl_train_status)
        layout.addStretch()

        # Calibration State Machine
        # Define target duration in seconds instead of frame ticks
        self.is_calibrating = False
        self.calib_states = [
            {"name": "REST (Prepare for Neutral)", "target": None, "duration": 3.0},
            {"name": "NEUTRAL (Resting)", "target": 0.0, "folder": "neutral_resting", "duration": 10.0},
            {"name": "NEUTRAL + Random Gaze (Hold 2s)", "target": 0.0, "folder": "neutral_random_gaze", "duration": 25.0},
            {"name": "REST (Prepare for Surprised)", "target": None, "duration": 3.0},
            {"name": "SURPRISED (Brows UP)", "target": 1.0, "folder": "surprised_brows_up", "duration": 10.0},
            {"name": "SURPRISED + Random Gaze (Hold 2s)", "target": 1.0, "folder": "surprised_brows_up_random_gaze", "duration": 25.0},
            {"name": "REST (Prepare for Lower Eyebrow)", "target": None, "duration": 3.0},
            {"name": "LOWER EYEBROW (Frown)", "target": -1.0, "folder": "frown_brows_down", "duration": 10.0},
            {"name": "FROWN + Random Gaze (Hold 2s)", "target": -1.0, "folder": "frown_brows_down_random_gaze", "duration": 25.0},
            {"name": "REST (Prepare for Sad)", "target": None, "duration": 3.0},
            {"name": "SAD (Inner Brows UP)", "target": 0.5, "folder": "sad_inner_brows_up", "duration": 10.0},
            {"name": "SAD INNER + Random Gaze (Hold 2s)", "target": 0.5, "folder": "sad_inner_brows_up_random_gaze", "duration": 25.0},
            {"name": "REST (Prepare for Smile)", "target": None, "duration": 3.0},
            {"name": "SMILE (Outer Brows DOWN)", "target": -0.5, "folder": "smile_outer_brows_down", "duration": 10.0},
            {"name": "SMILE OUTER + Random Gaze (Hold 2s)", "target": -0.5, "folder": "smile_outer_brows_down_random_gaze", "duration": 25.0}
        ]
        self.calib_idx = 0
        self.calib_start_time = 0.0

    def setup_console_tab(self):
        layout = QVBoxLayout(self.tab_console)
        self.txt_console = QPlainTextEdit()
        self.txt_console.setReadOnly(True)
        self.txt_console.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.txt_console.document().setMaximumBlockCount(2000)
        self.txt_console.setObjectName("console-text")
        mono = QFont("Consolas", 10)
        mono.setStyleHint(QFont.Monospace)
        self.txt_console.setFont(mono)
        layout.addWidget(self.txt_console)

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
        # Capture current raw EMA values to zero them out
        try:
            if self.ema_left.value is None or self.ema_right.value is None:
                self._show_error_dialog("Warning", "No tracking data yet.\nStart the streams and wait for values before setting baseline.")
                return
            self.offset_l = self.ema_left.value
            self.offset_r = self.ema_right.value
            QMessageBox.information(self, "Recalibrated", f"New Neutral Offsets:\nLeft: {self.offset_l:.2f}\nRight: {self.offset_r:.2f}")
        except Exception as e:
            self._show_error_dialog("Error", f"Failed to set baseline:\n{e}")

    def reset_offsets_zero(self):
        self.offset_l = 0.0
        self.offset_r = 0.0
        self.auto_offset_l = 0.0
        self.auto_offset_r = 0.0
        self.brow_history_l.clear()
        self.brow_history_r.clear()
        self.lbl_auto_status.setText("Status: Offsets reset to 0.0")
        self.lbl_auto_status.setStyleSheet("color: #888;")

    def update_dataset_status(self):
        self.lbl_dataset_status.setText(f"  |  Current Saved Dataset: {len(self.recorded_frames)} images")

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
        if QMessageBox.question(self, "Confirm", "Are you sure you want to delete all collected Eyebrow calibration images and CSVs?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
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
            QMessageBox.information(self, "Cleared", "Eyebrow calibration data cleared.")
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
                QMessageBox.warning(self, "Camera Error", "Select a camera device before starting the stream.")
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
                self.btn_osc.setText("Stop OSC Sender")
                self.btn_osc.setProperty("class", "danger-btn")
                self._refresh_button_style(self.btn_osc)
                self.txt_ip.setEnabled(False)
                self.txt_port.setEnabled(False)
            except Exception as e:
                QMessageBox.warning(self, "OSC Error", f"Could not create OSC client:\n{e}")
        else:
            # Disconnect
            self.osc_enabled = False
            self.osc_client = None
            
            self.btn_osc.setText("Start OSC Sender")
            self.btn_osc.setProperty("class", "success-btn")
            self._refresh_button_style(self.btn_osc)
            self.txt_ip.setEnabled(True)
            self.txt_port.setEnabled(True)

    def browse_weights(self):
        options = QFileDialog.Options()
        start_dir = self._get_model_dir(self.use_lr_models)
        if self.use_lr_models:
            left_path, _ = QFileDialog.getOpenFileName(self, "Select LEFT Eyebrow Weights", start_dir, "PyTorch Models (*.pth *.pt);;All Files (*)", options=options)
            if not left_path:
                return
            right_path, _ = QFileDialog.getOpenFileName(self, "Select RIGHT Eyebrow Weights", start_dir, "PyTorch Models (*.pth *.pt);;All Files (*)", options=options)
            if not right_path:
                return
            success = self.load_weights_lr(left_path, right_path)
            if success:
                self.lbl_current_model.setText(f"L/R Loaded: {os.path.basename(left_path)} | {os.path.basename(right_path)}")
                QMessageBox.information(self, "Success", "L/R eyebrow weights loaded successfully.")
            else:
                QMessageBox.critical(self, "Error", "Failed to load L/R weights.")
        else:
            file_name, _ = QFileDialog.getOpenFileName(self, "Select Eyebrow Weights", start_dir, "PyTorch Models (*.pth *.pt);;All Files (*)", options=options)
            if file_name:
                success = self.load_weights(file_name)
                if success:
                    self.lbl_current_model.setText(os.path.basename(file_name))
                    self._update_setting("last_model_path", file_name)
                    QMessageBox.information(self, "Success", "Eyebrow weights loaded successfully.")
                else:
                    QMessageBox.critical(self, "Error", "Failed to load weights. Incompatible model architecture.")
                
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
            
        expressions = [s['name'] for s in self.calib_states if s['target'] is not None]
        msg = "The following expressions will be recorded for 10 seconds each:\n\n"
        msg += "\n".join(expressions)
        msg += "\n\nMake sure your headset is firmly positioned."
        
        reply = QMessageBox.information(self, "Calibration Sequence", msg, QMessageBox.Ok | QMessageBox.Cancel)
        if reply == QMessageBox.Cancel:
            return

        self.btn_start_seq.setEnabled(False)
        self.btn_stop_seq.setEnabled(True)
        self.is_calibrating = True
        self.calib_idx = 0
        self.calib_start_time = time.time()
        self.calib_start_count = len(self.recorded_frames)
        self.lbl_seq_instruction.setText(f"Get Ready for: {self.calib_states[0]['name']}...")

    def _reset_seq_text(self):
        if not self.is_calibrating:
            self.lbl_seq_instruction.setText("Ready")
            self.lbl_seq_instruction.setStyleSheet("color: #111; margin-top: 20px; margin-bottom: 20px;")

    def stop_calibration_sequence(self):
        self.is_calibrating = False
        self.btn_start_seq.setEnabled(True)
        self.btn_stop_seq.setEnabled(False)
        self.lbl_seq_instruction.setText("Calibration Stopped. Discarding partial data...")
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
                frame_l_bgr, frame_r_bgr = self._split_combined_frame(combined)
            
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

        # Automatic Calibration Logic (UI Ticks regardless of new frames)
        if self.tabs.currentIndex() == 1 and self.is_calibrating and self.is_connected:
            state = self.calib_states[self.calib_idx]
            elapsed = curr_time - self.calib_start_time
            seconds_left = max(0, int(state['duration'] - elapsed) + 1)
            
            # Update UI
            if state['target'] is None:
                self.lbl_seq_instruction.setStyleSheet("color: #eb9534; margin-top: 20px; margin-bottom: 20px;")
                self.lbl_seq_instruction.setText(f"{state['name']}\n... {seconds_left}s ...")
            else:
                self.lbl_seq_instruction.setStyleSheet("color: #d32f2f; margin-top: 20px; margin-bottom: 20px;")
                self.lbl_seq_instruction.setText(f"HOLD: {state['name']}\nRecording... {seconds_left}s")
                
                # Save Frame only if it's an active recording state and genuinely new
                if is_new_frame and frame_l_bgr is not None and frame_r_bgr is not None:
                    folder_name = state.get('folder', 'unknown_folder')
                    self.save_calibration_frame(state['target'], folder_name, frame_l_bgr, frame_r_bgr)
            
            if elapsed >= state['duration']:
                self.calib_idx += 1
                if self.calib_idx >= len(self.calib_states):
                    # Finished
                    self.is_calibrating = False
                    self.lbl_seq_instruction.setText("Calibration Complete!\nYou can now bake the model.")
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
            # If nothing has changed natively in OpenCV, skip redundant workload
            if not is_new_frame:
                if not self.chk_manual.isChecked():
                    return
            
            tensor_l, tensor_r = None, None
            try:
                # Convert BGR OpenCV frames to grayscale
                gray_l = cv2.cvtColor(frame_l_bgr, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame_r_bgr, cv2.COLOR_BGR2GRAY)
                
                # Push grayscale arrays directly to GPU tensors
                tensor_l = torch.from_numpy(gray_l).unsqueeze(0).to(self.device)
                tensor_r = torch.from_numpy(gray_r).unsqueeze(0).to(self.device)
                
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
                
                fps_text = f"Capture FPS: {int(self.current_fps)}"
                self.lbl_l_fps.setText(fps_text)
                self.lbl_r_fps.setText(fps_text)
                
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
            elif self.is_connected and tensor_l is not None and tensor_r is not None:
                try:
                    import torch.nn.functional as F
                    
                    # Perform all crops/flips natively on the NVIDIA GPU!
                    h_l, w_l = tensor_l.shape[1], tensor_l.shape[2]
                    
                    crop_l = tensor_l[:, :int(h_l*0.4), int(w_l*0.15):int(w_l*0.85)]
                    crop_r = torch.flip(tensor_r, dims=[2])[:, :int(h_l*0.4), int(w_l*0.15):int(w_l*0.85)]
                    
                    from torchvision.transforms import functional as TF
                    
                    # Hardware bilinear interpolation with anti-aliasing to precisely match PIL training features
                    crop_l = TF.resize(crop_l, size=[64, 64], interpolation=TF.InterpolationMode.BILINEAR, antialias=True).unsqueeze(0).float()
                    crop_r = TF.resize(crop_r, size=[64, 64], interpolation=TF.InterpolationMode.BILINEAR, antialias=True).unsqueeze(0).float()
                    
                    # Normalize [0-255] natively to [-1.0, 1.0] for PyTorch and ensure hardware dispatch
                    batch = (torch.cat([crop_l, crop_r], dim=0) / 127.5 - 1.0).to(self.device)
                    
                    with torch.no_grad():
                        # Eyebrow Regressor
                        if self.use_lr_models:
                            self.model_left.eval()
                            self.model_right.eval()
                            out_l_t = torch.clamp(self.model_left(batch[0:1]), -1.0, 1.0)
                            out_r_t = torch.clamp(self.model_right(batch[1:2]), -1.0, 1.0)
                        else:
                            self.model.eval()
                            out_all = torch.clamp(self.model(batch), -1.0, 1.0)
                            out_l_t = out_all[0:1]
                            out_r_t = out_all[1:2]

                        # Outputs are [brow, inner, outer]
                        if out_l_t.dim() >= 2 and out_l_t.shape[1] >= 3:
                            raw_brow_l = out_l_t[0][0].item()
                            raw_inner_l = out_l_t[0][1].item()
                            raw_outer_l = out_l_t[0][2].item()
                        else:
                            raw_brow_l = out_l_t[0][0].item() if out_l_t.numel() > 0 else 0.0
                            raw_inner_l = raw_brow_l
                            raw_outer_l = raw_brow_l

                        if out_r_t.dim() >= 2 and out_r_t.shape[1] >= 3:
                            raw_brow_r = out_r_t[0][0].item()
                            raw_inner_r = out_r_t[0][1].item()
                            raw_outer_r = out_r_t[0][2].item()
                        else:
                            raw_brow_r = out_r_t[0][0].item() if out_r_t.numel() > 0 else 0.0
                            raw_inner_r = raw_brow_r
                            raw_outer_r = raw_brow_r

                        # If loaded weights are legacy (no inner/outer), mirror brow outputs
                        if self.use_lr_models:
                            if not self.model_left_has_inner_outer:
                                raw_inner_l = raw_brow_l
                                raw_outer_l = raw_brow_l
                            if not self.model_right_has_inner_outer:
                                raw_inner_r = raw_brow_r
                                raw_outer_r = raw_brow_r
                        else:
                            if not self.model_main_has_inner_outer:
                                raw_inner_l = raw_brow_l
                                raw_outer_l = raw_brow_l
                                raw_inner_r = raw_brow_r
                                raw_outer_r = raw_brow_r
                                
                    # --- AUTO BASELINE CORRECTION LOGIC (1D Statistical Variance) ---
                    self.brow_history_l.append(raw_brow_l)
                    self.brow_history_r.append(raw_brow_r)
                    if len(self.brow_history_l) > 60: self.brow_history_l.pop(0)
                    if len(self.brow_history_r) > 60: self.brow_history_r.pop(0)
                    
                    if self.chk_auto_baseline.isChecked() and len(self.brow_history_l) == 60:
                        import statistics
                        var_l = statistics.variance(self.brow_history_l)
                        var_r = statistics.variance(self.brow_history_r)
                        
                        VAR_THRESHOLD = 0.0005
                        T_MAX = (self.slider_alpha.value() / 100.0) * 0.1 # Map 0-100 slider to 0.0-0.10 max lerp
                        
                        var_max = max(var_l, var_r)
                        
                        if var_max < VAR_THRESHOLD:
                            stability = 1.0 - (var_max / VAR_THRESHOLD)
                            stability = max(0.0, min(1.0, stability))
                            t = T_MAX * (stability ** 2)
                            
                            # The new baseline is simply the mean of the stable 1-second window
                            new_baseline_l = sum(self.brow_history_l) / 60.0
                            new_baseline_r = sum(self.brow_history_r) / 60.0
                            
                            lerp = lambda a, b, t: a + (b - a) * t
                            
                            # Use lerp to slowly pull the auto_offset towards the new resting baseline 
                            self.auto_offset_l = lerp(self.auto_offset_l, new_baseline_l, t)
                            self.auto_offset_r = lerp(self.auto_offset_r, new_baseline_r, t)
                            
                            self.lbl_auto_status.setText(f"Status: Locked onto resting baseline (var={var_max:.5f})")
                            self.lbl_auto_status.setStyleSheet("color: #4CAF50;")
                        else:
                            t = 0.0
                            self.lbl_auto_status.setText(f"Status: Tracking expressions (var={var_max:.5f})")
                            self.lbl_auto_status.setStyleSheet("color: #eb9534;")
                            
                        # Apply corrective offset 
                        raw_brow_l -= self.auto_offset_l
                        raw_brow_r -= self.auto_offset_r
                    
                    # Apply standard Static Offsets and EMA Filter
                    alpha = max(0.01, 1.0 - (self.slider_smooth.value() / 100.0))
                    self.ema_left.alpha = alpha
                    self.ema_right.alpha = alpha
                    self.ema_inner_left.alpha = alpha
                    self.ema_inner_right.alpha = alpha
                    self.ema_outer_left.alpha = alpha
                    self.ema_outer_right.alpha = alpha
                    
                    out_l = self.ema_left.update(raw_brow_l) - self.offset_l
                    out_r = self.ema_right.update(raw_brow_r) - self.offset_r

                    inner_l = self.ema_inner_left.update(raw_inner_l)
                    inner_r = self.ema_inner_right.update(raw_inner_r)
                    outer_l = self.ema_outer_left.update(raw_outer_l)
                    outer_r = self.ema_outer_right.update(raw_outer_r)
                    
                    # Clamp to domain mapping
                    out_l = max(min(out_l, 1.0), -1.0)
                    out_r = max(min(out_r, 1.0), -1.0)
                    inner_l = max(min(inner_l, 1.0), -1.0)
                    inner_r = max(min(inner_r, 1.0), -1.0)
                    outer_l = max(min(outer_l, 1.0), -1.0)
                    outer_r = max(min(outer_r, 1.0), -1.0)
                    
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

            if self.osc_debug_enabled and hasattr(self, "osc_debug_widget"):
                ordered = [(self.osc_param_labels.get(k, k), self.osc_param_values.get(k, 0.0)) for k in self.osc_param_order]
                self.osc_debug_widget.set_data(ordered)
                if hasattr(self, "values_row_widget"):
                    ordered_vals = [(self.osc_param_labels.get(k, k), self.osc_param_values.get(k, 0.0)) for k in self.osc_param_order]
                    self.values_row_widget.set_data(ordered_vals)

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
                    self.btn_osc.setText("Start OSC Sender")
                    self.btn_osc.setProperty("class", "success-btn"); self.btn_osc.style().unpolish(self.btn_osc); self.btn_osc.style().polish(self.btn_osc)
                    self.txt_ip.setEnabled(True)
                    self.txt_port.setEnabled(True)

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
        model_dir = self._get_model_dir(self.use_lr_models)
        self.thread = TrainingThread(data_dir, train_csv, val_csv, model_dir, use_lr_models=self.use_lr_models, parent=self)
        self.thread.progress.connect(self.update_training_status)
        self.thread.finished.connect(self.training_finished)
        self.thread.start()
        
    def _set_training_buttons(self, enabled):
        self.btn_train.setEnabled(enabled)
        self.btn_train_with_path.setEnabled(enabled)

    def _refresh_button_style(self, btn):
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def _get_model_dir(self, use_lr=None):
        if use_lr is None:
            use_lr = self.use_lr_models
        appdata = os.getenv("APPDATA")
        if appdata:
            base_dir = Path(appdata) / "VREyebrowTracker"
        else:
            base_dir = Path("data")
        subdir = "models_lr" if use_lr else "models_single"
        model_dir = base_dir / subdir
        model_dir.mkdir(parents=True, exist_ok=True)
        return str(model_dir)

    def _apply_deadzone_boost(self, x):
        dz = self.slider_deadzone.value() / 100.0
        boost = self.slider_boost.value() / 100.0
        if dz >= 1.0:
            return 0.0
        if abs(x) < dz:
            return 0.0
        sign = 1.0 if x >= 0 else -1.0
        y = (abs(x) - dz) / (1.0 - dz)
        y = y * boost
        return max(-1.0, min(1.0, sign * y))

    def _apply_deadzone_boost_param(self, x, key):
        if key in getattr(self, "param_deadzone_sliders", {}):
            dz = self.param_deadzone_sliders[key].value() / 100.0
            boost_pos = self.param_boost_pos_sliders[key].value() / 100.0
            boost_neg = self.param_boost_neg_sliders[key].value() / 100.0
        else:
            dz = self.slider_deadzone.value() / 100.0
            boost_pos = self.slider_boost_pos.value() / 100.0
            boost_neg = self.slider_boost_neg.value() / 100.0
        if dz >= 1.0:
            return 0.0
        if abs(x) < dz:
            return 0.0
        sign = 1.0 if x >= 0 else -1.0
        y = (abs(x) - dz) / (1.0 - dz)
        y = y * (boost_pos if sign >= 0 else boost_neg)
        return max(-1.0, min(1.0, sign * y))

    def _toggle_param_osc(self, key, state):
        self.osc_param_enabled[key] = (state == Qt.Checked)
        self._update_setting(f"osc_enable_{key}", self.osc_param_enabled[key])

    def _toggle_osc_debug(self, state):
        self.osc_debug_enabled = (state == Qt.Checked)
        if hasattr(self, "osc_debug_widget"):
            if self.osc_debug_enabled:
                ordered = [(self.osc_param_labels.get(k, k), self.osc_param_values.get(k, 0.0)) for k in self.osc_param_order]
                self.osc_debug_widget.set_data(ordered)
                if hasattr(self, "values_row_widget"):
                    ordered_vals = [(self.osc_param_labels.get(k, k), self.osc_param_values.get(k, 0.0)) for k in self.osc_param_order]
                    self.values_row_widget.set_data(ordered_vals)
        if hasattr(self, "graph_stack"):
            self.graph_stack.setCurrentIndex(1 if self.osc_debug_enabled else 0)
        if hasattr(self, "graph_stack_widget"):
            if self.osc_debug_enabled:
                self.graph_stack_widget.setMaximumHeight(16777215)
            else:
                self.graph_stack_widget.setMaximumHeight(220)

    def _toggle_lr_models(self, state):
        self.use_lr_models = (state == Qt.Checked)
        if self.use_lr_models:
            self.lbl_current_model.setText("Eyebrow: L/R Separate (Beta)")
        else:
            if self.current_model_path and os.path.exists(self.current_model_path):
                self.lbl_current_model.setText(os.path.basename(self.current_model_path))
            else:
                self.lbl_current_model.setText("Eyebrow: None Loaded")

    def _compute_osc_values(self, brow_l, brow_r, inner_l, inner_r, outer_l, outer_r):
        return {
            "BrowExpressionLeft": float(brow_l),
            "BrowExpressionRight": float(brow_r),
            "BrowInnerUpLeft": float(inner_l),
            "BrowInnerUpRight": float(inner_r),
            "BrowOuterUpLeft": float(outer_l),
            "BrowOuterUpRight": float(outer_r),
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

    def training_finished(self, new_model_path):
        self._set_training_buttons(True)
        if new_model_path:
            try:
                if isinstance(new_model_path, (list, tuple)) and len(new_model_path) == 2:
                    lpath, rpath = new_model_path
                    if self.load_weights_lr(lpath, rpath):
                        self.lbl_current_model.setText(f"L/R Loaded: {os.path.basename(lpath)} | {os.path.basename(rpath)} (Newly Trained!)")
                else:
                    self.load_weights(new_model_path)
                    self.lbl_current_model.setText(f"{new_model_path} (Newly Trained!)")
            except:
                pass

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
