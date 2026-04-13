import PyInstaller.__main__
import os
import re
import shutil
from pathlib import Path

VERSION_FILE = "VERSION.txt"
GUI_FILE = "gui.py"
VERSION_RE = re.compile(r'^APP_VERSION\s*=\s*["\']([^"\']+)["\']')

def _read_version():
    env_ver = os.getenv("VREYEBROW_VERSION", "").strip()
    if env_ver:
        return env_ver
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "r", encoding="utf-8") as f:
            ver = f.read().strip()
            if ver:
                return ver
    return "0.0.0"

def _set_gui_version(version):
    if not os.path.exists(GUI_FILE):
        return False, ""
    path = Path(GUI_FILE)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    changed = False
    for i, line in enumerate(lines):
        if VERSION_RE.match(line.strip()):
            lines[i] = f'APP_VERSION = "{version}"'
            changed = True
            break
    if not changed:
        return False, text
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True, text

def build_executable():
    print("Preparing to build VR Eyebrow Tracker executable...")

    version = _read_version()
    updated, original_text = _set_gui_version(version)
    if updated:
        print(f"Set APP_VERSION to {version}")
    else:
        print("Warning: Failed to set APP_VERSION in gui.py")

    # 1. Clean previous builds
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")

    print("Running PyInstaller...")

    # PyInstaller Arguments — ONNX Runtime + DirectML inference
    try:
        PyInstaller.__main__.run([
            'inference.py',
            '--name=VREyebrowTracker',
            '--onefile',
            '--windowed',
            '--icon=NONE',
            '--log-level=WARN',

        '--hidden-import=onnxruntime',
        '--hidden-import=cv2',
        '--hidden-import=PIL',
        '--hidden-import=numpy',

        # Exclude heavy/unneeded packages
        '--exclude-module=torch',
        '--exclude-module=torchvision',
        '--exclude-module=torchaudio',
        '--exclude-module=tensorflow',
        '--exclude-module=tensorflow_core',
        '--exclude-module=tensorflow_estimator',
        '--exclude-module=tensorflow_io_gcs_filesystem',
        '--exclude-module=keras',
        '--exclude-module=matplotlib',
        '--exclude-module=tkinter',
        '--exclude-module=scipy',
        '--exclude-module=sklearn',
        '--exclude-module=setuptools',
        ])
    finally:
        if updated and original_text:
            Path(GUI_FILE).write_text(original_text, encoding="utf-8")
            print("Restored APP_VERSION in gui.py")

    print("\n===============================")
    print("Build Complete!")
    print("Your executable is located at: dist/VREyebrowTracker.exe")
    print("Note: Place 'eyebrow_model.onnx' in the same folder as the .exe!")
    print("===============================\n")

if __name__ == "__main__":
    build_executable()
