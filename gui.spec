# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
import os

def add_runtime_dll(name):
    sys_root = os.environ.get("SystemRoot", r"C:\Windows")
    path = os.path.join(sys_root, "System32", name)
    if os.path.exists(path):
        return (path, ".")
    return None

datas = []
binaries = []
hiddenimports = []

# PyQt5 (ensure all Qt DLLs and plugins are bundled, including platforms/qwindows.dll)
tmp_ret = collect_all('PyQt5')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ONNX Runtime (DirectML GPU inference)
tmp_ret = collect_all('onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# OpenCV
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Pandas (for dataset management)
tmp_ret = collect_all('pandas')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Requests (for update checks)
tmp_ret = collect_all('requests')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Copy onnxruntime DLLs to root level (DLL search path workaround)
ort_capi = os.path.join(os.path.dirname(os.path.abspath('gui.py')), 'venv_gpu', 'Lib', 'site-packages', 'onnxruntime', 'capi')
for dll_name in ('onnxruntime.dll', 'onnxruntime_providers_shared.dll', 'DirectML.dll'):
    dll_path = os.path.join(ort_capi, dll_name)
    if os.path.exists(dll_path):
        binaries.append((dll_path, '.'))

# Bundle training scripts (used by external Python subprocess for baking)
for script in ['train.py', 'model.py', 'dataset.py', 'onnx_inference.py', 'export_eyebrow_onnx.py']:
    if os.path.exists(script):
        datas.append((script, '.'))

# Bundle the default ONNX model and version file
if os.path.exists('eyebrow_model.onnx'):
    datas += [('eyebrow_model.onnx', '.'), ('eyebrow_model.onnx', '_internal')]
datas += [('VERSION.txt', '.'), ('VERSION.txt', '_internal')]
if os.path.exists('uninstall.bat'):
    datas += [('uninstall.bat', '.')]

for dll in [
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "concrt140.dll",
    "vcomp140.dll",
    "vcomp140_1.dll",
]:
    item = add_runtime_dll(dll)
    if item:
        binaries.append(item)

# Exclude torch entirely — training uses external Python subprocess
EXCLUDES = [
    'torch', 'torchvision', 'torchaudio', 'caffe2',
    'onnx.reference', 'onnx.backend',
    'sympy', 'mpmath',
    'onnxruntime.transformers', 'onnxruntime.quantization',
    'onnxruntime.tools',
    'tensorflow', 'keras',
    'matplotlib', 'scipy', 'sklearn',
    'IPython', 'jupyter', 'notebook',
]

# Filter torch from collected artifacts
datas = [(s, d) for s, d in datas if 'torch' not in s.lower()]
binaries = [(s, d) for s, d in binaries if 'torch' not in s.lower()]
hiddenimports = [h for h in hiddenimports if 'torch' not in h.lower()]

a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthook_ort.py'],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)

# Post-analysis: also filter torch from auto-detected binaries/datas
a.binaries = [b for b in a.binaries if 'torch' not in (b[1] or '').lower()]
a.datas = [d for d in a.datas if 'torch' not in d[0].lower()]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VR Eyebrow Tracker',
    icon='app_icon.ico',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='gui',
)
