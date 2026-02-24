import os

with open('c:\\Users\\kakao\\.gemini\\antigravity\\playground\\vast-meteor\\vr_eyebrow\\gui.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Make a backup!
with open('c:\\Users\\kakao\\.gemini\\antigravity\\playground\\vast-meteor\\vr_eyebrow\\gui_backup.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Replace imports
content = content.replace(
    'from PyQt5.QtGui import QImage, QPixmap, QFont',
    'from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor'
)

# Add is_dark_mode
content = content.replace(
    '        self.is_connected = False',
    '        self.is_dark_mode = True\\n        self.is_connected = False'
)

# Invoke apply_theme
content = content.replace(
    '        self.init_ui()',
    '        self.init_ui()\\n        self.apply_theme()'
)

# Replace top bar connect button and add theme toggle
top_bar_old = '''        # Top Bar (Connection)
        top_bar = QHBoxLayout()
        self.btn_connect = QPushButton("Connect to BrokenEye")
        self.btn_connect.setStyleSheet("""
            QPushButton { background-color: #2D68DB; color: white; font-weight: bold; padding: 10px; border-radius: 4px; font-size: 14px;}
            QPushButton:hover { background-color: #1F54B5; }
        """)
        self.btn_connect.clicked.connect(self.toggle_connection)
        top_bar.addWidget(self.btn_connect)
        main_layout.addLayout(top_bar)'''

top_bar_new = '''        # Top Bar (Connection)
        top_bar = QHBoxLayout()
        self.btn_connect = QPushButton("Connect to BrokenEye")
        self.btn_connect.setProperty("class", "primary-btn")
        self.btn_connect.clicked.connect(self.toggle_connection)
        top_bar.addWidget(self.btn_connect)
        
        self.btn_theme = QPushButton("☀️ Light Mode")
        self.btn_theme.setProperty("class", "theme-btn")
        self.btn_theme.clicked.connect(self.toggle_theme)
        top_bar.addWidget(self.btn_theme)
        
        main_layout.addLayout(top_bar)'''
content = content.replace(top_bar_old, top_bar_new)

# Camera labels
cam_l_old = 'self.left_img_label.setStyleSheet("background-color: #101010; border: 1px solid #CCC; color: white;")'
cam_l_new = 'self.left_img_label.setProperty("class", "cam-label")'
content = content.replace(cam_l_old, cam_l_new)

cam_r_old = 'self.right_img_label.setStyleSheet("background-color: #101010; border: 1px solid #CCC; color: white;")'
cam_r_new = 'self.right_img_label.setProperty("class", "cam-label")'
content = content.replace(cam_r_old, cam_r_new)

# OSC Sender Button
btn_osc_old = 'self.btn_osc.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")'
btn_osc_new = 'self.btn_osc.setProperty("class", "success-btn")'
content = content.replace(btn_osc_old, btn_osc_new)

# btn_start_seq
btn_start_seq_old = '''        self.btn_start_seq.setStyleSheet("""
            QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 20px; font-size: 16px; border-radius: 8px;}
            QPushButton:hover { background-color: #d32f2f; }
        """)'''
btn_start_seq_new = '        self.btn_start_seq.setProperty("class", "primary-btn-danger")'
content = content.replace(btn_start_seq_old, btn_start_seq_new)

# btn_train
btn_train_old = 'self.btn_train.setStyleSheet("background-color: #673AB7; color: white; padding: 15px; font-weight: bold; font-size: 14px;")'
btn_train_new = 'self.btn_train.setProperty("class", "primary-btn-purple")'
content = content.replace(btn_train_old, btn_train_new)

# toggle_connection
toggle_conn_old1 = 'self.btn_connect.setStyleSheet("background-color: #E63946; color: white; font-weight: bold; padding: 10px; border-radius: 4px; font-size: 14px;")'
toggle_conn_new1 = 'self.btn_connect.setProperty("class", "primary-btn-danger"); self.btn_connect.style().unpolish(self.btn_connect); self.btn_connect.style().polish(self.btn_connect)'
content = content.replace(toggle_conn_old1, toggle_conn_new1)

toggle_conn_old2 = 'self.btn_connect.setStyleSheet("background-color: #2D68DB; color: white; font-weight: bold; padding: 10px; border-radius: 4px; font-size: 14px;")'
toggle_conn_new2 = 'self.btn_connect.setProperty("class", "primary-btn"); self.btn_connect.style().unpolish(self.btn_connect); self.btn_connect.style().polish(self.btn_connect)'
content = content.replace(toggle_conn_old2, toggle_conn_new2)

# toggle_osc
toggle_osc_old1 = 'self.btn_osc.setStyleSheet("background-color: #E63946; color: white; font-weight: bold; padding: 8px;")'
toggle_osc_new1 = 'self.btn_osc.setProperty("class", "danger-btn"); self.btn_osc.style().unpolish(self.btn_osc); self.btn_osc.style().polish(self.btn_osc)'
content = content.replace(toggle_osc_old1, toggle_osc_new1)

toggle_osc_old2 = 'self.btn_osc.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")'
toggle_osc_new2 = 'self.btn_osc.setProperty("class", "success-btn"); self.btn_osc.style().unpolish(self.btn_osc); self.btn_osc.style().polish(self.btn_osc)'
content = content.replace(toggle_osc_old2, toggle_osc_new2)

# Specific Label colors
lbl_l_fps_old = 'self.lbl_l_fps.setStyleSheet("color: #888; font-size: 11px; font-weight: bold;")'
lbl_l_fps_new = 'self.lbl_l_fps.setProperty("class", "muted-label")'
content = content.replace(lbl_l_fps_old, lbl_l_fps_new)

lbl_r_fps_old = 'self.lbl_r_fps.setStyleSheet("color: #888; font-size: 11px; font-weight: bold;")'
lbl_r_fps_new = 'self.lbl_r_fps.setProperty("class", "muted-label")'
content = content.replace(lbl_r_fps_old, lbl_r_fps_new)

lbl_current_model_old = 'self.lbl_current_model.setStyleSheet("color: #666; font-size: 11px;")'
lbl_current_model_new = 'self.lbl_current_model.setProperty("class", "muted-label")'
content = content.replace(lbl_current_model_old, lbl_current_model_new)

lbl_osc_instr_old = 'lbl_osc_instr.setStyleSheet("font-size: 10px; color: #666;")'
lbl_osc_instr_new = 'lbl_osc_instr.setProperty("class", "muted-label")'
content = content.replace(lbl_osc_instr_old, lbl_osc_instr_new)

instr_old = 'instr.setStyleSheet("color: #444; font-size: 13px; font-weight: bold;")'
instr_new = 'instr.setProperty("class", "bold-label")'
content = content.replace(instr_old, instr_new)

lbl_train_old = 'lbl_train.setStyleSheet("margin-top: 20px;")'
lbl_train_new = 'lbl_train.setProperty("class", "header-label")'
content = content.replace(lbl_train_old, lbl_train_new)

info_lbl_old = 'info_lbl.setStyleSheet("color: #444; font-size: 14px; font-weight: bold;")'
info_lbl_new = 'info_lbl.setProperty("class", "header-label")'
content = content.replace(info_lbl_old, info_lbl_new)


# Insert apply_theme and toggle_theme before toggle_connection
theme_funcs = '''
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
        QLabel { background-color: transparent; }
        QPushButton { background-color: #333; color: white; border: 1px solid #555; border-radius: 4px; padding: 6px; }
        QPushButton:hover { background-color: #444; }
        
        QPushButton[class="primary-btn"] { background-color: #0078D7; color: white; font-weight: bold; padding: 10px; border: none; font-size: 14px;}
        QPushButton[class="primary-btn"]:hover { background-color: #005A9E; }
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
        
        QLabel[class="cam-label"] { background-color: #000000; border: 2px solid #444; border-radius: 8px; }
        QLabel[class="muted-label"] { color: #888; font-size: 11px; }
        QLabel[class="bold-label"] { color: #ccc; font-size: 13px; font-weight: bold; }
        QLabel[class="header-label"] { color: #ddd; font-size: 14px; font-weight: bold; margin-top: 20px; }
        
        QScrollArea { border: none; }
        QScrollBar:vertical { background: #1e1e1e; width: 12px; margin: 0px; }
        QScrollBar::handle:vertical { background: #444; min-height: 20px; border-radius: 6px; margin: 2px;}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
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
        QLabel { background-color: transparent; }
        QPushButton { background-color: #e0e0e0; color: black; border: 1px solid #bbb; border-radius: 4px; padding: 6px; }
        QPushButton:hover { background-color: #d0d0d0; }
        
        QPushButton[class="primary-btn"] { background-color: #0078D7; color: white; font-weight: bold; padding: 10px; border: none; font-size: 14px;}
        QPushButton[class="primary-btn"]:hover { background-color: #005A9E; }
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
        
        QLabel[class="cam-label"] { background-color: #000000; border: 2px solid #ccc; border-radius: 8px; }
        QLabel[class="muted-label"] { color: #555; font-size: 11px; }
        QLabel[class="bold-label"] { color: #333; font-size: 13px; font-weight: bold; }
        QLabel[class="header-label"] { color: #222; font-size: 14px; font-weight: bold; margin-top: 20px; }
        
        QScrollArea { border: none; background: transparent; }
        QScrollBar:vertical { background: #f5f5f5; width: 12px; margin: 0px; }
        QScrollBar::handle:vertical { background: #ccc; min-height: 20px; border-radius: 6px; margin: 2px;}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
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

    def toggle_connection(self):'''

content = content.replace("    def toggle_connection(self):", theme_funcs)

with open('c:\\Users\\kakao\\.gemini\\antigravity\\playground\\vast-meteor\\vr_eyebrow\\gui.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patching complete.")
