"""ONNX Runtime + DirectML inference for TinyBrowNet eyebrow tracking.

Drop-in replacement for PyTorch inference. Supports DirectML (Windows GPU),
CUDA, and CPU fallback. Preprocessing uses numpy/cv2 only — no torch required.
"""

import cv2
import numpy as np

import onnxruntime as ort


class HMDShiftTracker:
    """Detects HMD positional shift via phase correlation on stable image regions.

    Uses the bottom portion of the camera image (cheek/nose area) which is
    unaffected by eyebrow or eye expressions, to measure frame-to-frame
    pixel displacement caused by physical HMD movement.

    This enables input-level correction: the ROI crop position follows the
    HMD shift in real-time, so the model always sees the same skin patch
    regardless of headset position. Like optical image stabilization.
    """

    # Which portion of the image to use as anchor (bottom 40% = cheek/nose)
    ANCHOR_TOP_RATIO = 0.6
    # Maximum cumulative shift before auto-reset (pixels)
    MAX_SHIFT = 60
    # Minimum confidence to accept a phase correlation result
    MIN_CONFIDENCE = 0.03

    def __init__(self):
        self._prev = None
        self._hann = None
        self._shift = np.array([0.0, 0.0])  # accumulated (dx, dy) in pixels
        self._confidence = 0.0

    def update(self, gray):
        """Measure HMD shift from previous frame.

        Args:
            gray: Full camera grayscale image (H, W) uint8.

        Returns:
            (dx, dy) cumulative pixel shift since reset/init.
        """
        h, w = gray.shape[:2]
        anchor = gray[int(h * self.ANCHOR_TOP_RATIO):, :]
        anchor_f = anchor.astype(np.float64)

        if self._prev is None or anchor_f.shape != self._prev.shape:
            self._prev = anchor_f
            ah, aw = anchor_f.shape
            self._hann = cv2.createHanningWindow((aw, ah), cv2.CV_64F)
            return 0.0, 0.0

        (dx, dy), response = cv2.phaseCorrelate(self._prev, anchor_f, self._hann)
        self._confidence = self._confidence * 0.7 + response * 0.3

        if response > self.MIN_CONFIDENCE:
            self._shift[0] += dx
            self._shift[1] += dy
            self._prev = anchor_f

        # Auto-reset if shift grows too large (camera changed drastically)
        if np.linalg.norm(self._shift) > self.MAX_SHIFT:
            self.reset(gray)
            return 0.0, 0.0

        return float(self._shift[0]), float(self._shift[1])

    def reset(self, gray=None):
        """Reset tracking origin. Optionally set a new reference frame."""
        if gray is not None:
            h, w = gray.shape[:2]
            anchor = gray[int(h * self.ANCHOR_TOP_RATIO):, :]
            self._prev = anchor.astype(np.float64)
            ah, aw = self._prev.shape
            self._hann = cv2.createHanningWindow((aw, ah), cv2.CV_64F)
        else:
            self._prev = None
            self._hann = None
        self._shift[:] = 0
        self._confidence = 0.0

    @property
    def confidence(self):
        return self._confidence

    @property
    def shift_px(self):
        return float(self._shift[0]), float(self._shift[1])


def get_available_providers():
    """Return list of (label, provider_name) tuples for ONNX Runtime."""
    available = ort.get_available_providers()
    providers = []
    if "DmlExecutionProvider" in available:
        providers.append(("DirectML (GPU)", "DmlExecutionProvider"))
    if "CUDAExecutionProvider" in available:
        providers.append(("CUDA (GPU)", "CUDAExecutionProvider"))
    providers.append(("CPU", "CPUExecutionProvider"))
    return providers


class BrowNetONNX:
    """ONNX Runtime inference wrapper for TinyBrowNet eyebrow models.

    Loads a single .onnx file and runs batch inference.
    Preprocessing mirrors the PyTorch pipeline: ROI crop, resize 64x64,
    normalize grayscale [0-255] -> [-1, 1].
    """

    def __init__(self, model_path, provider=None):
        """
        Args:
            model_path: Path to .onnx model file.
            provider: ONNX execution provider name (e.g. "DmlExecutionProvider").
                      If None, auto-selects best available.
        """
        providers = self._resolve_providers(provider)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(str(model_path), opts, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

        # Detect output width from model metadata
        out_shape = self._session.get_outputs()[0].shape
        self._output_width = out_shape[-1] if len(out_shape) >= 2 else 3
        self._active_provider = self._session.get_providers()[0]

    @staticmethod
    def _resolve_providers(provider):
        available = ort.get_available_providers()
        if provider and provider in available:
            return [provider, "CPUExecutionProvider"]
        # Auto-select: DirectML > CUDA > CPU
        if "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @property
    def active_provider(self):
        return self._active_provider

    @property
    def output_width(self):
        return self._output_width

    @staticmethod
    def preprocess_crop(gray, preprocessed=False, shift_px=(0, 0)):
        """Crop ROI with HMD shift compensation, resize to 64x64.

        Args:
            gray: Grayscale numpy array (H, W), uint8.
            preprocessed: If True, skip ROI crop (already 64x64 aligned).
            shift_px: (dx, dy) pixel shift from HMDShiftTracker to compensate.

        Returns:
            numpy array (1, 1, 64, 64) float32 in [-1, 1].
        """
        if not preprocessed:
            h, w = gray.shape[:2]
            dx, dy = shift_px
            # Standard ROI: top 40%, sides 15%-85%
            # Shift crop to follow HMD movement (content moves opposite to HMD)
            top = max(0, min(h - 2, int(0 + dy)))
            bot = max(top + 1, min(h, int(h * 0.4 + dy)))
            left = max(0, min(w - 2, int(w * 0.15 + dx)))
            right = max(left + 1, min(w, int(w * 0.85 + dx)))
            gray = gray[top:bot, left:right]

        img = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR)
        # Normalize [0, 255] -> [-1, 1]
        arr = img.astype(np.float32) / 127.5 - 1.0
        return arr.reshape(1, 1, 64, 64)

    def predict_pair(self, gray_left, gray_right, preprocessed=False,
                     shift_l=(0, 0), shift_r=(0, 0)):
        """Run inference on a left/right eye pair.

        Args:
            gray_left: Left eye grayscale (H, W) uint8.
            gray_right: Right eye grayscale (H, W) uint8. Will be h-flipped.
            preprocessed: If True, images are already 64x64 aligned.
            shift_l: (dx, dy) HMD shift for left eye in pixels.
            shift_r: (dx, dy) HMD shift for right eye in pixels.

        Returns:
            (left_output, right_output) where each is [brow, inner, outer].
        """
        crop_l = self.preprocess_crop(gray_left, preprocessed, shift_px=shift_l)
        # Mirror right eye to match "left eye" training convention
        gray_r_flipped = cv2.flip(gray_right, 1)
        # Flip dx for mirrored image
        shift_r_flipped = (-shift_r[0], shift_r[1])
        crop_r = self.preprocess_crop(gray_r_flipped, preprocessed, shift_px=shift_r_flipped)

        batch = np.concatenate([crop_l, crop_r], axis=0)
        result = self._session.run(None, {self._input_name: batch})
        out = result[0]  # shape (2, output_width)

        out_l = np.clip(out[0], -1.0, 1.0).tolist()
        out_r = np.clip(out[1], -1.0, 1.0).tolist()

        # Pad to 3 outputs if legacy single-output model
        while len(out_l) < 3:
            out_l.append(out_l[0])
        while len(out_r) < 3:
            out_r.append(out_r[0])

        return out_l[:3], out_r[:3]

    def predict_single(self, gray, preprocessed=False, shift_px=(0, 0)):
        """Run inference on a single eye image.

        Args:
            gray: Grayscale (H, W) uint8.
            preprocessed: If True, skip ROI crop.
            shift_px: (dx, dy) HMD shift in pixels.

        Returns:
            [brow, inner, outer] clipped to [-1, 1].
        """
        crop = self.preprocess_crop(gray, preprocessed, shift_px=shift_px)
        result = self._session.run(None, {self._input_name: crop})
        out = np.clip(result[0][0], -1.0, 1.0).tolist()
        while len(out) < 3:
            out.append(out[0])
        return out[:3]


def export_pth_to_onnx(pth_path, onnx_path, batch_size=2, opset=17):
    """Helper: export a PyTorch .pth checkpoint to ONNX.

    Requires torch + the model module. Used for one-time conversion.
    """
    import torch
    from model import TinyBrowNet
    from export_eyebrow_onnx import ExportWrapper, load_state_dict_compat
    from pathlib import Path

    model = TinyBrowNet()
    mirror = load_state_dict_compat(model, Path(pth_path))
    wrapper = ExportWrapper(model.eval(), mirror).eval()

    dummy = torch.randn(batch_size, 1, 64, 64, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,
        )
    print(f"Exported: {pth_path} -> {onnx_path}")


if __name__ == "__main__":
    print("Available ONNX providers:", get_available_providers())
    print("Use export_pth_to_onnx() to convert .pth -> .onnx")
