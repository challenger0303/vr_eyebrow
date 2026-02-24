# VR Eyebrow Tracker (PyTorch)

A lightweight, high-FPS PyTorch pipeline to estimate eyebrow Up/Down movements using indirect internal IR eye-tracking cameras (e.g., BrokenEye). Since the eyebrow itself is outside the IR camera's field of view, this method focuses on the upper-eyelid and skin deformation.

## Core Concepts

1. **ROI Extraction**: It crops the top 40% of the 600x600 IR image. This effectively removes the pupil, eyelashes, and cheek that add noise or false correlations, forcing the CNN to look purely at the skin wrinkles and stretch above the eye.
2. **Tiny Architecture**: Utilizes `TinyBrowNet`, a custom 4-layer CNN predicting a continuous value from `[-1.0 (Frown) to +1.0 (Surprised)]`. This allows inference execution well over 90 FPS on a low-end GPU.
3. **Temporal Smoothing**: Uses an Exponential Moving Average (EMA) to prevent jitter without the latency cost of LSTMs or RNNs.

## Data Collection Matrix (CRITICAL)

Because the eyebrow tracking relies on skin deformation near the eye, it is extremely easy for a neural network to confuse "Eye blinking" with "Eyebrow dropping" or "Eyes wide" with "Eyebrow raising".

To prevent this, you **MUST** collect training frames enforcing all combinations of eye states and brow states. By doing this, the CNN learns to isolate the brow deformation, ignoring the open/closed state of the eye itself.

Record ~30-60 seconds of video for each of these 6 states:

| Brow State | Eye State | Target Label |
| :--- | :--- | :--- |
| Neutral | Open | `0.0` |
| Neutral | Closed | `0.0` |
| UP (Surprised) | Open | `1.0` |
| UP (Surprised) | Closed | `1.0` |
| DOWN (Frown) | Open | `-1.0` |
| DOWN (Frown) | Closed | `-1.0` |

Extract these videos into individual frames and log their corresponding `Target Label` into a `train.csv` and `val.csv` file.

## Usage

1. **Prepare Data**:
   Place your images in `data/images/` and create `data/train.csv` and `data/val.csv` following this format:
   ```csv
   filename,label
   neutral_open_001.png,0.0
   surprised_closed_014.png,1.0
   frown_open_040.png,-1.0
   ```

2. **Train Model**:
   ```bash
   python train.py
   ```
   This will train the TinyBrowNet and save the best weights to `tinybrownet_best.pth`.

3. **Run Inference**:
   Replace the `camera_id=0` in `inference.py` with your Python code that fetches the BrokenEye 600x600 grayscale IR byte stream.
   ```bash
   python inference.py
   ```
   *Note: In `inference.py`, there is a commented out line showing where to hook up the output to a Python OSC library (`python-osc`) to send the smoothed prediction directly to VRChat.*
