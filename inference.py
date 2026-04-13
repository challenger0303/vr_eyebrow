import cv2
import numpy as np
import time
from onnx_inference import BrowNetONNX


class EMARegressor:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value


class PredictiveInterpolator:
    """Simple EMA smoother. Drop-in compatible with the old EMARegressor interface."""

    def __init__(self, smooth=0.3):
        self.smooth = smooth
        self._value = None

    @property
    def value(self):
        return self._value

    def update(self, new_val):
        if self._value is None:
            self._value = new_val
        else:
            self._value = self.smooth * new_val + (1.0 - self.smooth) * self._value
        return self._value

    def extrapolate(self):
        return self._value if self._value is not None else 0.0


def setup_transform():
    """Legacy compatibility stub — preprocessing is now in BrowNetONNX."""
    return None


def crop_roi(image_pil):
    """Crop top 40% ROI from a PIL image (legacy compatibility)."""
    w, h = image_pil.size
    crop_box = (int(w * 0.15), 0, int(w * 0.85), int(h * 0.4))
    return image_pil.crop(crop_box)


def run_dual_inference(model_path="eyebrow_model.onnx",
                       left_url="http://127.0.0.1:5555/eye/left",
                       right_url="http://127.0.0.1:5555/eye/right"):

    model = BrowNetONNX(model_path)
    print(f"Running ONNX inference on {model.active_provider}")

    ema_left = EMARegressor(alpha=0.3)
    ema_right = EMARegressor(alpha=0.3)

    cap_left = cv2.VideoCapture(left_url)
    cap_right = cv2.VideoCapture(right_url)
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Could not open one or more streams.")
        return

    print("Starting dual-eye evaluation. Press 'q' to quit.")

    while True:
        start_time = time.time()
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

        if not ret_l or not ret_r:
            break

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        out_l, out_r = model.predict_pair(gray_l, gray_r)
        raw_l = out_l[0]  # brow
        raw_r = out_r[0]  # brow

        smooth_l = ema_left.update(raw_l)
        smooth_r = ema_right.update(raw_r)

        fps = 1.0 / max(time.time() - start_time, 1e-6)

        # Build UI
        h_l, w_l = gray_l.shape
        crop_l_vis = gray_l[0:int(h_l*0.4), int(w_l*0.15):int(w_l*0.85)]
        crop_r_vis = cv2.flip(gray_r, 1)
        h_r, w_r = crop_r_vis.shape
        crop_r_vis = crop_r_vis[0:int(h_r*0.4), int(w_r*0.15):int(w_r*0.85)]

        ui_l = cv2.cvtColor(cv2.resize(crop_l_vis, (200, 100)), cv2.COLOR_GRAY2BGR)
        ui_r = cv2.cvtColor(cv2.resize(crop_r_vis, (200, 100)), cv2.COLOR_GRAY2BGR)

        ui_l = draw_tracker_ui(ui_l, smooth_l, "L-Brow")
        ui_r = draw_tracker_ui(ui_r, smooth_r, "R-Brow")

        combined = cv2.hconcat([ui_l, ui_r])
        cv2.putText(combined, f"FPS: {fps:.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow('Dual Eyebrow Tracker (ONNX)', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


def draw_tracker_ui(display_img, prediction, label_text):
    display_img = cv2.resize(display_img, (400, 200))

    bar_height = 150
    offset = int(-prediction * (bar_height // 2))

    cv2.rectangle(display_img, (350, 25), (370, 175), (0, 0, 0), -1)
    cv2.line(display_img, (340, 100), (380, 100), (255, 255, 255), 2)

    color = (0, 255, 0) if prediction > 0 else (0, 0, 255)
    cv2.circle(display_img, (360, 100 + offset), 8, color, -1)

    cv2.putText(display_img, f"{label_text}: {prediction:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return display_img


if __name__ == "__main__":
    run_dual_inference()
