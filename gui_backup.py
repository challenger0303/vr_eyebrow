import sys
import os
import time
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import requests
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QCheckBox, 
                             QLineEdit, QFrame, QGroupBox, QStyleFactory, QTabWidget, 
                             QProgressBar, QFileDialog, QMessageBox, QSlider, QTableWidget, QTableWidgetItem, QHeaderView,
                             QScrollArea, QGridLayout)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
from torchvision import transforms
# Proxy imports
import socket
import json
import struct
from pythonosc.udp_client import SimpleUDPClient

from model import TinyBrowNet
from inference import EMARegressor, setup_transform, crop_roi
from train import train_model

class CameraThread(QThread):
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.latest_jpeg = None
        self.running = False
        
    def run(self):
        self.running = True
        try:
            r = requests.get(self.url, stream=True, timeout=5)
            
            # Extract the actual MJPEG boundary from HTTP headers
            content_type = r.headers.get("Content-Type", "")
            boundary_str = ""
            if "boundary=" in content_type:
                boundary_str = content_type.split("boundary=")[-1].strip()
                
            if not boundary_str:
                boundary = b'\r\n\r\n'
            else:
                boundary = b"--" + boundary_str.encode('utf-8')
                
            bytes_data = b''
            for chunk in r.iter_content(chunk_size=8192):
                if not self.running: break
                bytes_data += chunk
                
                while True:
                    a = bytes_data.find(boundary)
                    if a == -1: break
                    b = bytes_data.find(boundary, a + len(boundary))
                    if b == -1: break
                    
                    frame_data = bytes_data[a:b]
                    bytes_data = bytes_data[b:] # Keep the upcoming boundary
                    
                    jpg_start = frame_data.find(b'\xff\xd8')
                    jpg_end = frame_data.rfind(b'\xff\xd9')
                    
                    # Ensure valid JPEG
                    if jpg_start != -1 and jpg_end != -1 and jpg_end > jpg_start:
                        self.latest_jpeg = frame_data[jpg_start:jpg_end+2]
                        
                # Memory safety
                if len(bytes_data) > 1024 * 1024:
                    bytes_data = b''
        except Exception as e:
            print(f"Camera HTTP Error ({self.url}):", e)

    def stop(self):
        self.running = False
        self.wait()

class TrainingThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    
    def run(self):
        self.progress.emit("Starting PyTorch Training...")
        try:
            DATA_DIR = "./data/images/"
            TRAIN_CSV = "./data/train.csv"
            VAL_CSV = "./data/val.csv"
            self.progress.emit("Training in progress (check terminal for logs)...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = f"model_{timestamp}.pth"
            train_model(DATA_DIR, TRAIN_CSV, VAL_CSV, save_path=save_path)
            self.progress.emit(f"Training Complete! Saved '{save_path}'.")
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
            save_path = ""
        finally:
            self.finished.emit(save_path)

class BrokenEyeTCPProxyThread(QThread):
    osc_msg_signal = pyqtSignal(str, str, bool)
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
                    
                json_str = data.decode('utf-8')
                try:
                    payload = json.loads(json_str)
                    
                    # Intercept and Drop Eyebrow Values (Squeeze and Wide are mapped to Eyebrows)
                    blocked = False
                    for side in ["Left", "Right"]:
                        if side in payload:
                            if "Squeeze" in payload[side] and payload[side]["Squeeze"] != 0.0:
                                payload[side]["Squeeze"] = 0.0
                                blocked = True
                            if "Wide" in payload[side] and payload[side]["Wide"] != 0.0:
                                payload[side]["Wide"] = 0.0
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
        self.is_connected = False
        self.cam_left = None
        self.cam_right = None
        
        # OSC State
        self.osc_client = None
        self.osc_proxy = None
        self.osc_enabled = False
        self.osc_ip = "127.0.0.1"
        self.osc_port = 9000
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TinyBrowNet().to(self.device)
        self.current_model_path = "None Loaded"
            
        self.transform = setup_transform()
        self.ema_left = EMARegressor(alpha=0.3)
        self.ema_right = EMARegressor(alpha=0.3)
        
        self.current_fps = 0.0
        self.last_update_time = time.time()
        
        # Data Collection State
        self.recorded_frames = []
        self.data_dir = Path("data")
        self.images_dir = self.data_dir / "images"
        self.csv_path = self.data_dir / "train.csv"
        
        if self.csv_path.exists():
            try:
                self.recorded_frames = pd.read_csv(self.csv_path).to_dict('records')
            except: pass
            
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.val_csv_path = self.data_dir / "val.csv"
        
        # UI & Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
        self.osc_data_cache = {}
        self.observer_timer = QTimer(self)
        self.observer_timer.timeout.connect(self.update_osc_table)
        self.observer_timer.start(100) # 10 Hz UI refresh for OSC table
        
        self.init_ui()
        
    def load_weights(self, path):
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            self.current_model_path = path
            return True
        except:
            return False

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top Bar (Connection)
        top_bar = QHBoxLayout()
        self.btn_connect = QPushButton("Connect to BrokenEye")
        self.btn_connect.setStyleSheet("""
            QPushButton { background-color: #2D68DB; color: white; font-weight: bold; padding: 10px; border-radius: 4px; font-size: 14px;}
            QPushButton:hover { background-color: #1F54B5; }
        """)
        self.btn_connect.clicked.connect(self.toggle_connection)
        top_bar.addWidget(self.btn_connect)
        main_layout.addLayout(top_bar)
        
        # TABS
        self.tabs = QTabWidget()
        self.tab_tracker = QWidget()
        self.tab_calibration = QWidget()
        self.tab_observer = QWidget()
        
        self.tabs.addTab(self.tab_tracker, "1. Live Tracker & OSC")
        self.tabs.addTab(self.tab_calibration, "2. Calibration & Training")
        self.tabs.addTab(self.tab_observer, "3. OSC Observer")
        main_layout.addWidget(self.tabs)
        
        self.setup_tracker_tab()
        self.setup_calibration_tab()
        self.setup_observer_tab()
        
    def setup_tracker_tab(self):
        layout = QHBoxLayout(self.tab_tracker)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # ====================
        # Left Side (Cameras)
        # ====================
        eye_layout = QVBoxLayout()
        cam_row = QHBoxLayout()
        
        # Left Eye
        self.left_eye_box = QVBoxLayout()
        
        self.lbl_l_fps = QLabel("FPS: 0")
        self.lbl_l_fps.setAlignment(Qt.AlignLeft)
        self.lbl_l_fps.setStyleSheet("color: #888; font-size: 11px; font-weight: bold;")
        
        self.left_img_label = QLabel("Left Eye Stream")
        self.left_img_label.setFixedSize(250, 250)
        self.left_img_label.setStyleSheet("background-color: #101010; border: 1px solid #CCC; color: white;")
        self.left_img_label.setAlignment(Qt.AlignCenter)
        
        lbl_l = QLabel("Left Eye")
        lbl_l.setAlignment(Qt.AlignCenter)
        lbl_l.setFont(QFont("Arial", 11, QFont.Bold))
        
        self.lbl_l_brow = QLabel("Brow Slider: 0.00")
        self.lbl_l_brow.setAlignment(Qt.AlignCenter)
        
        self.left_eye_box.addWidget(self.lbl_l_fps)
        self.left_eye_box.addWidget(self.left_img_label)
        self.left_eye_box.addWidget(lbl_l)
        self.left_eye_box.addWidget(self.lbl_l_brow)
        
        # Right Eye
        self.right_eye_box = QVBoxLayout()
        
        self.lbl_r_fps = QLabel("FPS: 0")
        self.lbl_r_fps.setAlignment(Qt.AlignLeft)
        self.lbl_r_fps.setStyleSheet("color: #888; font-size: 11px; font-weight: bold;")
        
        self.right_img_label = QLabel("Right Eye Stream")
        self.right_img_label.setFixedSize(250, 250)
        self.right_img_label.setStyleSheet("background-color: #101010; border: 1px solid #CCC; color: white;")
        self.right_img_label.setAlignment(Qt.AlignCenter)
        
        lbl_r = QLabel("Right Eye")
        lbl_r.setAlignment(Qt.AlignCenter)
        lbl_r.setFont(QFont("Arial", 11, QFont.Bold))
        
        self.lbl_r_brow = QLabel("Brow Slider: 0.00")
        self.lbl_r_brow.setAlignment(Qt.AlignCenter)
        
        self.right_eye_box.addWidget(self.lbl_r_fps)
        self.right_eye_box.addWidget(self.right_img_label)
        self.right_eye_box.addWidget(lbl_r)
        self.right_eye_box.addWidget(self.lbl_r_brow)
        
        cam_row.addLayout(self.left_eye_box)
        cam_row.addLayout(self.right_eye_box)
        eye_layout.addLayout(cam_row)
        
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
        
        # Model Selection Group
        grp_model = QGroupBox("Model Configuration")
        grp_model_layout = QVBoxLayout()
        
        btn_load_model = QPushButton("Load Weights (.pth)")
        btn_load_model.clicked.connect(self.browse_weights)
        grp_model_layout.addWidget(btn_load_model)
        
        self.lbl_current_model = QLabel("None Loaded")
        self.lbl_current_model.setStyleSheet("color: #666; font-size: 11px;")
        grp_model_layout.addWidget(self.lbl_current_model)
        
        grp_model.setLayout(grp_model_layout)
        settings_panel.addWidget(grp_model)
        
        # OSC Server & Proxy Group
        grp_osc = QGroupBox("OSC Output & Filter Proxy")
        grp_osc_layout = QVBoxLayout()
        
        lbl_osc_instr = QLabel("AI Output Parameters:\n/avatar/parameters/FT/v2/BrowExpression...")
        lbl_osc_instr.setStyleSheet("font-size: 10px; color: #666;")
        grp_osc_layout.addWidget(lbl_osc_instr)
        
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("AI Output IP:"))
        self.txt_ip = QLineEdit("127.0.0.1")
        port_layout.addWidget(self.txt_ip)
        port_layout.addWidget(QLabel("Port (VRChat):"))
        self.txt_port = QLineEdit("9000")
        self.txt_port.setFixedWidth(50)
        port_layout.addWidget(self.txt_port)
        grp_osc_layout.addLayout(port_layout)
        
        self.chk_proxy = QCheckBox("Enable BrokenEye TCP Filter Proxy")
        self.chk_proxy.setChecked(True)
        self.chk_proxy.setToolTip("Intercepts TCP JSON data, zeroes 'Squeeze/Wide' (Eyebrow) params, and forwards the rest to VRCFT.")
        grp_osc_layout.addWidget(self.chk_proxy)
        
        proxy_layout = QHBoxLayout()
        proxy_layout.addWidget(QLabel("Proxy Listen Port:"))
        self.txt_proxy_listen = QLineEdit("5556")
        self.txt_proxy_listen.setFixedWidth(50)
        proxy_layout.addWidget(self.txt_proxy_listen)
        proxy_layout.addWidget(QLabel("Target Port (BrokenEye):"))
        self.txt_proxy_target = QLineEdit("5555")
        self.txt_proxy_target.setFixedWidth(50)
        proxy_layout.addWidget(self.txt_proxy_target)
        grp_osc_layout.addLayout(proxy_layout)
        
        self.btn_osc = QPushButton("Start OSC Sender")
        self.btn_osc.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.btn_osc.clicked.connect(self.toggle_osc)
        grp_osc_layout.addWidget(self.btn_osc)
        
        grp_osc.setLayout(grp_osc_layout)
        settings_panel.addWidget(grp_osc)
        
        # Manual Override Group
        grp_manual = QGroupBox("Manual Browser Override")
        grp_manual_layout = QVBoxLayout()
        
        self.chk_manual = QCheckBox("Enable Manual Sliders")
        grp_manual_layout.addWidget(self.chk_manual)
        
        slider_layout = QHBoxLayout()
        
        # Left Slider
        vbox_l = QVBoxLayout()
        vbox_l.addWidget(QLabel("Left"))
        self.slider_l = QSlider(Qt.Vertical)
        self.slider_l.setRange(-100, 100)
        self.slider_l.setValue(0)
        self.slider_l.setTickPosition(QSlider.TicksBothSides)
        self.slider_l.valueChanged.connect(self.snap_left_slider)
        vbox_l.addWidget(self.slider_l, alignment=Qt.AlignHCenter)
        slider_layout.addLayout(vbox_l)
        
        # Right Slider
        vbox_r = QVBoxLayout()
        vbox_r.addWidget(QLabel("Right"))
        self.slider_r = QSlider(Qt.Vertical)
        self.slider_r.setRange(-100, 100)
        self.slider_r.setValue(0)
        self.slider_r.setTickPosition(QSlider.TicksBothSides)
        self.slider_r.valueChanged.connect(self.snap_right_slider)
        vbox_r.addWidget(self.slider_r, alignment=Qt.AlignHCenter)
        slider_layout.addLayout(vbox_r)
        
        grp_manual_layout.addLayout(slider_layout)
        grp_manual.setLayout(grp_manual_layout)
        settings_panel.addWidget(grp_manual)
        
        # Sync Group
        grp_sync = QGroupBox("L/R Synchronization")
        grp_sync_layout =QVBoxLayout()
        self.lbl_sync_val = QLabel("Sync: 0%")
        grp_sync_layout.addWidget(self.lbl_sync_val)
        
        self.slider_sync = QSlider(Qt.Horizontal)
        self.slider_sync.setRange(0, 100)
        self.slider_sync.setValue(0)
        self.slider_sync.valueChanged.connect(lambda v: self.lbl_sync_val.setText(f"Sync: {v}%"))
        grp_sync_layout.addWidget(self.slider_sync)
        
        grp_sync.setLayout(grp_sync_layout)
        settings_panel.addWidget(grp_sync)
        
        settings_panel.addStretch()
        layout.addLayout(settings_panel, stretch=1)
        
    def setup_calibration_tab(self):
        layout = QVBoxLayout(self.tab_calibration)
        
        instr = QLabel("The manual buttons have been removed.\n"
                       "Click START, and follow the instructions on screen.\n"
                       "The system will automatically record your face for each expression.")
        instr.setStyleSheet("color: #444; font-size: 13px; font-weight: bold;")
        layout.addWidget(instr)
        
        # Automatic Guided Sequence UI
        seq_layout = QVBoxLayout()
        
        self.btn_start_seq = QPushButton("START GUIDED CALIBRATION")
        self.btn_start_seq.setStyleSheet("""
            QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 20px; font-size: 16px; border-radius: 8px;}
            QPushButton:hover { background-color: #d32f2f; }
        """)
        self.btn_start_seq.clicked.connect(self.start_calibration_sequence)
        seq_layout.addWidget(self.btn_start_seq)
        
        self.lbl_seq_instruction = QLabel("Ready")
        self.lbl_seq_instruction.setAlignment(Qt.AlignCenter)
        self.lbl_seq_instruction.setFont(QFont("Arial", 24, QFont.Bold))
        self.lbl_seq_instruction.setStyleSheet("color: #111; margin-top: 20px; margin-bottom: 20px;")
        seq_layout.addWidget(self.lbl_seq_instruction)
        
        layout.addLayout(seq_layout)
        
        # Training Section
        lbl_train = QLabel("Training Pipeline")
        lbl_train.setFont(QFont("Arial", 12, QFont.Bold))
        lbl_train.setStyleSheet("margin-top: 20px;")
        layout.addWidget(lbl_train)
        
        self.btn_train = QPushButton("BAKE MODEL (Start Training)")
        self.btn_train.setStyleSheet("background-color: #673AB7; color: white; padding: 15px; font-weight: bold; font-size: 14px;")
        self.btn_train.clicked.connect(self.start_training)
        layout.addWidget(self.btn_train)
        
        self.lbl_train_status = QLabel("Status: Idle")
        layout.addWidget(self.lbl_train_status)
        layout.addStretch()

        # Calibration State Machine
        # ~66 fps * 10s = 660 ticks for recording
        # ~66 fps * 3s = 200 ticks for resting
        self.is_calibrating = False
        self.calib_states = [
            {"name": "REST (Prepare for Neutral)", "target": None, "duration": 200},
            {"name": "NEUTRAL (Resting)", "target": 0.0, "duration": 660},
            {"name": "REST (Prepare for Surprised)", "target": None, "duration": 200},
            {"name": "SURPRISED (Brows UP)", "target": 1.0, "duration": 660},
            {"name": "REST (Prepare for Frown)", "target": None, "duration": 200},
            {"name": "FROWN / ANGRY (Brows DOWN)", "target": -1.0, "duration": 660},
            {"name": "REST (Prepare for Sad)", "target": None, "duration": 200},
            {"name": "SAD (Inner Brows UP)", "target": 0.5, "duration": 660}
        ]
        self.calib_idx = 0
        self.calib_ticks_remaining = 0

    def setup_observer_tab(self):
        layout = QVBoxLayout(self.tab_observer)
        
        info_lbl = QLabel("VRCFT Unified Expressions (Live Backend Traffic)")
        info_lbl.setStyleSheet("color: #444; font-size: 14px; font-weight: bold;")
        layout.addWidget(info_lbl)
        
        # We need a scroll area because there are many parameters
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_widget)
        
        self.scroll_area.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll_area)
        
        # We use a dictionary to store references to the UI progress bars so we can update them rapidly
        self.ui_bars = {}

        # Target parameters to track visually (similar to VRCFT screenshot list)
        self.target_keys = ["PupilDiameterMm", "GazeX", "GazeY", "Openness", "Squeeze", "Wide"]
        
        # Prepopulate the UI so it isn't blank on launch
        row_idx = 0
        for side, col in [("Left", 0), ("Right", 2)]:
            hlbl = QLabel(f"--- {side} Eye ---")
            hlbl.setStyleSheet("font-weight: bold;")
            self.scroll_layout.addWidget(hlbl, row_idx, col, 1, 2, Qt.AlignCenter)
            self.ui_bars[f"H_{side}"] = hlbl
            
            temp_row = row_idx + 1
            for key in self.target_keys:
                path = f"{side}.{key}"
                lbl, bar = self._create_bar(path)
                self.scroll_layout.addWidget(lbl, temp_row, col)
                self.scroll_layout.addWidget(bar, temp_row, col+1)
                self.ui_bars[path] = bar
                temp_row += 1

    def clear_osc_table(self):
        self.osc_data_cache.clear()
        # Reset bars to 0
        for key, widget in self.ui_bars.items():
            if isinstance(widget, QProgressBar):
                widget.setValue(0)

    def cache_osc_message(self, address, val_str, is_blocked):
        # In TCP mode, 'address' is the raw JSON string payload dumped from the proxy thread
        self.osc_data_cache["latest_json"] = address
        self.update_osc_table()
        
    def _create_bar(self, path):
        # Helper to dynamically build UI bars
        label = QLabel(path.replace("Left.", "").replace("Right.", ""))
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        label.setFixedWidth(120)
        
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setFixedHeight(12)
        
        # Distinct color rules
        if "Squeeze" in path or "Wide" in path:
            bar.setStyleSheet("QProgressBar::chunk { background-color: #f44336; }") # Red for intercepted AI parameters
        else:
            bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }") # Green for forwarded natively
            
        return label, bar

    def update_osc_table(self):
        # Update Table visually
        if self.tabs.currentIndex() != 2:
            return 
            
        json_str = self.osc_data_cache.get("latest_json", "{}")
        try:
            payload = json.loads(json_str)
        except:
            return

        # The target keys and UI bars are already instantiated in setup_observer_tab
        for side in ["Left", "Right"]:
            if side in payload:
                for key in self.target_keys:
                    path = f"{side}.{key}"
                    
                    if path in self.ui_bars:
                        val = payload[side].get(key, 0.0)
                        
                        # Normalize ranges to 0-100%
                        if key == "PupilDiameterMm": pct = int(min(max(val / 8.0, 0.0), 1.0) * 100) # Assuming ~8mm max
                        elif key == "GazeX" or key == "GazeY": pct = int(((val + 1.0) / 2.0) * 100) # Remap -1 -> 1 to 0 -> 100
                        else: pct = int(min(max(val, 0.0), 1.0) * 100)
                        
                        self.ui_bars[path].setValue(pct)

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

    def toggle_connection(self):
        if not self.is_connected:
            self.cam_left = CameraThread("http://127.0.0.1:5555/eye/left")
            self.cam_right = CameraThread("http://127.0.0.1:5555/eye/right")
            
            self.cam_left.start()
            self.cam_right.start()
            
            self.is_connected = True
            self.btn_connect.setText("Disconnect from device")
            self.btn_connect.setStyleSheet("background-color: #E63946; color: white; font-weight: bold; padding: 10px; border-radius: 4px; font-size: 14px;")
            self.timer.start(15) # ~66 fps
        else:
            self.is_connected = False
            self.btn_connect.setText("Connect to BrokenEye")
            self.btn_connect.setStyleSheet("background-color: #2D68DB; color: white; font-weight: bold; padding: 10px; border-radius: 4px; font-size: 14px;")
            self.timer.stop()
            if self.cam_left: self.cam_left.stop()
            if self.cam_right: self.cam_right.stop()
            self.left_img_label.clear()
            self.right_img_label.clear()

    def toggle_osc(self):
        if not self.osc_enabled:
            # Connect
            ip = self.txt_ip.text()
            port = int(self.txt_port.text())
            try:
                self.osc_client = SimpleUDPClient(ip, port)
                
                # Start Proxy if enabled
                if self.chk_proxy.isChecked():
                    p_listen = int(self.txt_proxy_listen.text())
                    p_target = int(self.txt_proxy_target.text())
                    self.osc_proxy = BrokenEyeTCPProxyThread(listen_ip="127.0.0.1", listen_port=p_listen, target_ip="127.0.0.1", target_port=p_target)
                    self.osc_proxy.osc_msg_signal.connect(self.cache_osc_message)
                    self.osc_proxy.start()
                    
                self.osc_enabled = True
                self.btn_osc.setText("Stop OSC Sender & Proxy")
                self.btn_osc.setStyleSheet("background-color: #E63946; color: white; font-weight: bold; padding: 8px;")
                self.txt_ip.setEnabled(False)
                self.txt_port.setEnabled(False)
                self.chk_proxy.setEnabled(False)
                self.txt_proxy_listen.setEnabled(False)
                self.txt_proxy_target.setEnabled(False)
            except Exception as e:
                QMessageBox.warning(self, "OSC Error", f"Could not create OSC client:\n{e}")
        else:
            # Disconnect
            self.osc_enabled = False
            self.osc_client = None
            
            # Stop proxy if running
            if self.osc_proxy:
                self.osc_proxy.stop()
                self.osc_proxy = None
                
            self.btn_osc.setText("Start OSC Sender & Proxy")
            self.btn_osc.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
            self.txt_ip.setEnabled(True)
            self.txt_port.setEnabled(True)
            self.chk_proxy.setEnabled(True)
            self.txt_proxy_listen.setEnabled(True)
            self.txt_proxy_target.setEnabled(True)

    def browse_weights(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Model Weights", "", "PyTorch Models (*.pth *.pt);;All Files (*)", options=options)
        if file_name:
            success = self.load_weights(file_name)
            if success:
                self.lbl_current_model.setText(os.path.basename(file_name))
                QMessageBox.information(self, "Success", "Weights loaded successfully.")
            else:
                QMessageBox.critical(self, "Error", "Failed to load weights. Incompatible model architecture.")
            
    def save_calibration_frame(self, target_val, label_str, frame_l, frame_r):
        uid = len(self.recorded_frames)
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        
        name_l = f"{label_str}_l_{uid:05d}.jpg"
        cv2.imwrite(str(self.images_dir / name_l), gray_l)
        self.recorded_frames.append({"filename": name_l, "label": target_val})
        
        name_r = f"{label_str}_r_{uid:05d}.jpg"
        cv2.imwrite(str(self.images_dir / name_r), cv2.flip(gray_r, 1))
        self.recorded_frames.append({"filename": name_r, "label": target_val})
        
        df = pd.DataFrame(self.recorded_frames)
        df = df.sample(frac=1).reset_index(drop=True)
        train_size = int(len(df) * 0.8)
        df.iloc[:train_size].to_csv(self.csv_path, index=False)
        df.iloc[train_size:].to_csv(self.val_csv_path, index=False)

    def start_calibration_sequence(self):
        if not self.is_connected:
            QMessageBox.warning(self, "Warning", "Please connect to BrokenEye first!")
            return
            
        self.btn_start_seq.setEnabled(False)
        self.is_calibrating = True
        self.calib_idx = 0
        self.calib_ticks_remaining = self.calib_states[0]['duration']
        self.lbl_seq_instruction.setText(f"Get Ready for: {self.calib_states[0]['name']}...")

    def update_frame(self):
        # We handle Manual mode even if offline
        offline_manual = not self.is_connected and self.chk_manual.isChecked() and self.tabs.currentIndex() == 0

        if not self.is_connected and not offline_manual:
            return
            
        if self.is_connected:
            jpg_l = self.cam_left.latest_jpeg
            jpg_r = self.cam_right.latest_jpeg
            if jpg_l is None or jpg_r is None: return
            
            if getattr(self, 'last_jpg_l', None) is jpg_l and getattr(self, 'last_jpg_r', None) is jpg_r:
                if not self.chk_manual.isChecked():
                    return
            
            self.last_jpg_l = jpg_l
            self.last_jpg_r = jpg_r
            
            tensor_l, tensor_r = None, None
            try:
                # Hardware accelerated JPEG decode to GPU tensor
                tensor_l = torchvision.io.decode_jpeg(torch.frombuffer(bytearray(jpg_l), dtype=torch.uint8), mode=torchvision.io.ImageReadMode.GRAY, device=self.device)
                tensor_r = torchvision.io.decode_jpeg(torch.frombuffer(bytearray(jpg_r), dtype=torch.uint8), mode=torchvision.io.ImageReadMode.GRAY, device=self.device)
                
                # Retrieve grayscale array for GUI compatibility
                gray_l = tensor_l.squeeze(0).cpu().numpy()
                gray_r = tensor_r.squeeze(0).cpu().numpy()
                
                # Mock BGR frame to preserve backward compatibility with data collection logic
                frame_l = cv2.cvtColor(gray_l, cv2.COLOR_GRAY2BGR)
                frame_r = cv2.cvtColor(gray_r, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                # Catch decoding assertions quietly
                return
            
            curr_time = time.time()
            dt = curr_time - getattr(self, 'last_update_time', curr_time)
            self.last_update_time = curr_time
            if dt > 0:
                self.current_fps = self.current_fps * 0.9 + (1.0 / dt) * 0.1
        else:
            frame_l, frame_r = None, None
        
        # Automatic Calibration Logic
        if self.tabs.currentIndex() == 1 and self.is_calibrating and self.is_connected:
            state = self.calib_states[self.calib_idx]
            
            # Update UI
            pct = 100 - int((self.calib_ticks_remaining / state['duration']) * 100)
            
            if state['target'] is None:
                self.lbl_seq_instruction.setStyleSheet("color: #eb9534; margin-top: 20px; margin-bottom: 20px;")
                self.lbl_seq_instruction.setText(f"{state['name']}\n... {int(self.calib_ticks_remaining/66)+1}s ...")
            else:
                self.lbl_seq_instruction.setStyleSheet("color: #d32f2f; margin-top: 20px; margin-bottom: 20px;")
                self.lbl_seq_instruction.setText(f"HOLD: {state['name']}\nRecording... {pct}%")
                
                # Save Frame only if it's an active recording state
                if state['target'] == 0.0:  lbl="neu"
                elif state['target'] == 1.0: lbl="up"
                elif state['target'] == -1.0: lbl="dwn"
                else: lbl="sad"
                self.save_calibration_frame(state['target'], lbl, frame_l, frame_r)
            
            self.calib_ticks_remaining -= 1
            if self.calib_ticks_remaining <= 0:
                self.calib_idx += 1
                if self.calib_idx >= len(self.calib_states):
                    # Finished
                    self.is_calibrating = False
                    self.lbl_seq_instruction.setText("Calibration Complete!\nYou can now bake the model.")
                    self.btn_start_seq.setEnabled(True)
                else:
                    self.calib_ticks_remaining = self.calib_states[self.calib_idx]['duration']

        # Inference Logic
        if self.is_connected and frame_l is not None and frame_r is not None:
            if self.tabs.currentIndex() == 0:
                # Displays (Always update visuals)
                h, w = gray_l.shape
                
                fps_text = f"Capture FPS: {int(self.current_fps)}"
                self.lbl_l_fps.setText(fps_text)
                self.lbl_r_fps.setText(fps_text)
                
                self.left_img_label.setPixmap(QPixmap.fromImage(QImage(gray_l.data, w, h, w, QImage.Format_Grayscale8)).scaled(250, 250, Qt.KeepAspectRatio))
                self.right_img_label.setPixmap(QPixmap.fromImage(QImage(gray_r.data, w, h, w, QImage.Format_Grayscale8)).scaled(250, 250, Qt.KeepAspectRatio))
            
            # Decide where the values come from
            if self.chk_manual.isChecked():
                # Read from UI sliders (-100 to 100 -> -1.0 to 1.0)
                out_l = self.slider_l.value() / 100.0
                out_r = self.slider_r.value() / 100.0
                lbl_suffix = " (MANUAL)"
            elif self.is_connected and tensor_l is not None and tensor_r is not None:
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
                    # Clamp the linear outputs to the slider domain since we removed tanh() from the CNN
                    outputs = torch.clamp(self.model(batch), -1.0, 1.0)
                    if outputs.shape == (2, 1):
                        raw_l, raw_r = outputs[0].item(), outputs[1].item()
                    else: raw_l, raw_r = 0.0, 0.0
                    
                out_l = self.ema_left.update(raw_l)
                out_r = self.ema_right.update(raw_r)
                lbl_suffix = ""
            else:
                out_l = 0.0
                out_r = 0.0
                lbl_suffix = ""
                
            # Apply Synchronization Blending
            sync_factor = self.slider_sync.value() / 100.0
            if sync_factor > 0.0:
                avg_val = (out_l + out_r) / 2.0
                out_l = out_l * (1.0 - sync_factor) + (avg_val * sync_factor)
                out_r = out_r * (1.0 - sync_factor) + (avg_val * sync_factor)
                
            if self.tabs.currentIndex() == 0:
                self.lbl_l_brow.setText(f"Brow Slider: {out_l:.2f}{lbl_suffix}")
                self.lbl_r_brow.setText(f"Brow Slider: {out_r:.2f}{lbl_suffix}")
                
            # Broadcast OSC if enabled
            if self.osc_enabled and self.osc_client:
                self.osc_client.send_message("/avatar/parameters/FT/v2/BrowExpressionLeft", float(out_l))
                self.osc_client.send_message("/avatar/parameters/FT/v2/BrowExpressionRight", float(out_r))

    def start_training(self):
        if len(self.recorded_frames) < 10:
            QMessageBox.warning(self, "Warning", "Not enough data! Please record frames first.")
            return
            
        self.btn_train.setEnabled(False)
        self.thread = TrainingThread()
        self.thread.progress.connect(self.update_training_status)
        self.thread.finished.connect(self.training_finished)
        self.thread.start()
        
    def update_training_status(self, msg):
        self.lbl_train_status.setText(f"Status: {msg}")

    # Update training finish to accept string
    def training_finished(self, new_model_path):
        self.btn_train.setEnabled(True)
        try:
            self.load_weights(new_model_path)
            self.lbl_current_model.setText(f"{new_model_path} (Newly Trained!)")
        except: pass

if __name__ == '__main__':
    if hasattr(Qt, 'AA_EnableHighDpiScaling'): QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    ex = VREyebrowTrackerGUI()
    ex.show()
    sys.exit(app.exec_())
