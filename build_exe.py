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
        return False
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
        return False
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True

def build_executable():
    print("Preparing to build VR Eyebrow Tracker executable...")

    version = _read_version()
    if _set_gui_version(version):
        print(f"Set APP_VERSION to {version}")
    else:
        print("Warning: Failed to set APP_VERSION in gui.py")
    
    # 1. Clean previous builds
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
        
    print("Running PyInstaller...")
    
    # PyInstaller Arguments
    PyInstaller.__main__.run([
        'inference.py',
        '--name=VREyebrowTracker',
        '--onefile',                   # Package into a single .exe
        '--windowed',                  # Do not open a background terminal window
        '--icon=NONE',                 # (Optional: specify an .ico file path here later)
        '--log-level=WARN',            # Reduce noise; set DEBUG to diagnose issues

        # PyTorch often needs these hidden imports explicitly declared
        '--hidden-import=torch',
        '--hidden-import=torchvision',
        '--hidden-import=cv2',
        '--hidden-import=PIL',

        # --- Exclude modules that are NOT needed and cause DLL conflicts ---
        # TensorFlow/Keras DLLs conflict with PyTorch CUDA DLLs at runtime.
        # These get dragged in as indirect dependencies - force them out.
        '--exclude-module=tensorflow',
        '--exclude-module=tensorflow_core',
        '--exclude-module=tensorflow_estimator',
        '--exclude-module=tensorflow_io_gcs_filesystem',
        '--exclude-module=keras',
        '--exclude-module=keras.src',

        # Other heavy/unneeded packages
        '--exclude-module=matplotlib',
        '--exclude-module=tkinter',
        '--exclude-module=pandas',
        '--exclude-module=scipy',
        '--exclude-module=sklearn',
        '--exclude-module=setuptools',
        '--exclude-module=torch._dynamo',
        '--exclude-module=torch.distributed',
        '--exclude-module=torch.testing',
    ])
    
    print("\n===============================")
    print("Build Complete!")
    print("Your executable is located at: dist/VREyebrowTracker.exe")
    print("Note: To run the .exe, ensure 'tinybrownet_best.pth' is in the exact same folder as the .exe!")
    print("===============================\n")

if __name__ == "__main__":
    build_executable()
