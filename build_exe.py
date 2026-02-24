import PyInstaller.__main__
import os
import shutil

def build_executable():
    print("Preparing to build VR Eyebrow Tracker executable...")
    
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
