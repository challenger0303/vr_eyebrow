# VR Eyebrow Tracker

A lightweight eyebrow tracking application for VR headsets. Uses ONNX Runtime + DirectML for GPU-accelerated inference on any GPU (NVIDIA, AMD, Intel). Works with BrokenEye, Pimax, Bigscreen Beyond, and DIY eye tracking cameras.

## Features

- **GPU Inference via DirectML** - Works on NVIDIA, AMD, and Intel GPUs
- **TinyBrowNet CNN** - Custom 4-layer CNN (35K params), inference in <1ms
- **HMD Shift Compensation** - Real-time image-level tracking via phase correlation
- **Power Curve Response** - Configurable deadzone curve to prevent accidental expressions
- **MJPEG Camera Sharing** - Share camera feed with Baballonia for simultaneous eye + eyebrow tracking
- **Guided Dataset Capture** - Frame-count based recording with automatic train/val split
- **In-App Model Baking** - Train and export models without leaving the GUI
- **OSC Output** - VRChat-compatible face tracking parameters (VRCFT/FT v2)

## Quick Start

### From Source
```
run.bat
```
This will:
1. Find or create a Python virtual environment
2. Install dependencies (onnxruntime-directml, PyTorch, etc.)
3. Launch the GUI

### From Built Executable
Download the latest ZIP from Releases, extract it, and run **`Launch.bat`**.

`Launch.bat` automatically installs the Visual C++ 2015-2022 Redistributable (x64) if it's missing, which is required by ONNX Runtime and PyQt5.

If you run `VR Eyebrow Tracker.exe` directly and see errors like:
- `DLL load failed while importing QtWidgets`
- `DLL load failed while importing onnxruntime_pybind11_state`

...run **`install_vc_redist.bat`** once, then try `Launch.bat` again.

To remove all app data (training Python env, settings), run **`uninstall.bat`**.

## OSC Parameters

The following parameters are sent to VRChat via OSC:

| Parameter | Address | Range | Description |
|-----------|---------|-------|-------------|
| BrowExpressionLeft | `/avatar/parameters/FT/v2/BrowExpressionLeft` | -1 to +1 | Left eyebrow (full range) |
| BrowExpressionRight | `/avatar/parameters/FT/v2/BrowExpressionRight` | -1 to +1 | Right eyebrow (full range) |
| BrowUpLeft | `/avatar/parameters/FT/v2/BrowUpLeft` | 0 to 1 | Left eyebrow raised |
| BrowUpRight | `/avatar/parameters/FT/v2/BrowUpRight` | 0 to 1 | Right eyebrow raised |
| BrowDownLeft | `/avatar/parameters/FT/v2/BrowDownLeft` | 0 to 1 | Left eyebrow lowered |
| BrowDownRight | `/avatar/parameters/FT/v2/BrowDownRight` | 0 to 1 | Right eyebrow lowered |
| BrowUp | `/avatar/parameters/FT/v2/BrowUp` | 0 to 1 | Both eyebrows raised (average) |
| BrowDown | `/avatar/parameters/FT/v2/BrowDown` | 0 to 1 | Both eyebrows lowered (average) |

Each parameter can be toggled on/off in the Debug panel.

## Tabs

### Tracker
- Live camera preview (left/right eye)
- Start/Stop camera streams (URL or device)
- OSC sender toggle
- Model loading (.onnx or .pth with auto-conversion)
- Signal smoothing, L/R symmetry matching
- Headset recenter (manual + auto-drift follow)
- Inference graph
- Debug panel: per-parameter curve/boost tuning with live preview, OSC parameter toggles, manual override

### Training
- Guided dataset capture (frame-count based)
- Bake model from captured data or external folder
- Embedded console with training progress
- Auto ONNX export after training

### Settings
- Compute device selection (DirectML GPU / CPU)
- HMD profile (Pimax/Varjo, Bigscreen Beyond, DIY)
- Light/Dark theme
- OSC IP/port configuration
- Baballonia camera sharing (MJPEG server port)
- GitHub update checker

## Camera Sharing with Baballonia

For HMDs with a single shared camera (e.g., Bigscreen Beyond):

1. Select **Bigscreen Beyond** or **DIY** as HMD profile
2. Enable **Share camera via MJPEG** in the Tracker tab
3. Copy the address shown (e.g., `http://localhost:8085/mjpeg`)
4. Paste it into Baballonia's Camera Address field

This app grabs the camera and re-broadcasts via MJPEG, so Baballonia can receive frames without a camera lock conflict.

## Architecture

```
Camera (BrokenEye / DirectShow / MJPEG)
  |
  v
HMD Shift Tracker (phase correlation on stable image region)
  |
  v
ROI Crop (top 40%, sides 15-85%, shift-compensated)
  |
  v
TinyBrowNet ONNX (64x64 grayscale -> [brow, inner, outer])
  |  via ONNX Runtime + DirectML (GPU)
  |
  v
Auto-Baseline Correction (time-based, hysteresis)
  |
  v
EMA Smoothing -> Power Curve -> Symmetry Calibration
  |
  v
OSC Output (8 parameters to VRChat)
```

## Data Collection

The guided capture collects ~5,250 frames per session across 5 expressions:
- Neutral (resting + random gaze)
- Surprised (brows up + random gaze)
- Frown (brows down + random gaze)
- Sad (inner brows up + random gaze)
- Smile (outer brows down + random gaze)

## Requirements

- Windows 10/11
- DirectX 12 compatible GPU (any vendor)
- Python 3.10+ (for source/training)
- BrokenEye or compatible eye tracking camera

## Build

```
build.bat
```
Produces a ~407 MB distributable in `dist/gui/`.

Training requires PyTorch (installed in venv, not bundled in the exe). The built exe automatically finds a nearby venv with PyTorch for model baking.

## License

See LICENSE file.
