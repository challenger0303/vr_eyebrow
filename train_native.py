"""Native ridge regression trainer for eyebrow model. No PyTorch required.

Same approach as Baballonia's NativeLinearEyebrowTrainerBackend:
  1. Load 64x64 grayscale images
  2. Average pool 4x4 -> 16x16 = 256 features
  3. Ridge regression (closed-form solution, no iteration)
  4. Export ONNX model: AveragePool -> Flatten -> Gemm

Works with numpy only. Runs in <5 seconds.
"""

import csv
import os
import numpy as np
from pathlib import Path

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


POOL_SIZE = 4
INPUT_W, INPUT_H = 64, 64
POOLED_W, POOLED_H = INPUT_W // POOL_SIZE, INPUT_H // POOL_SIZE
FEATURE_COUNT = POOLED_W * POOLED_H  # 256
OUTPUT_COUNT = 3  # brow, inner, outer
RIDGE_LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0]


def load_image_gray(path):
    """Load image as 64x64 grayscale float [-1, 1]."""
    from PIL import Image
    img = Image.open(path).convert('L')
    if img.size != (INPUT_W, INPUT_H):
        img = img.resize((INPUT_W, INPUT_H), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    return arr / 127.5 - 1.0


def extract_features(img):
    """Average pool 4x4 -> 256 features."""
    h, w = img.shape
    pooled = img.reshape(h // POOL_SIZE, POOL_SIZE, w // POOL_SIZE, POOL_SIZE).mean(axis=(1, 3))
    return pooled.flatten()


def load_dataset(csv_path, img_dir, preprocessed=False):
    """Load dataset from CSV + image directory. Returns (features, targets)."""
    if not os.path.exists(csv_path):
        return np.empty((0, FEATURE_COUNT)), np.empty((0, OUTPUT_COUNT))

    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(img_dir, row['filename'])
            if not os.path.exists(img_path):
                continue

            brow = float(row.get('brow', row.get('label', 0)))
            inner = float(row.get('inner', brow))
            outer = float(row.get('outer', brow))

            try:
                img = load_image_gray(img_path)
                features = extract_features(img)
                rows.append((features, [brow, inner, outer]))
            except Exception:
                continue

    if not rows:
        return np.empty((0, FEATURE_COUNT)), np.empty((0, OUTPUT_COUNT))

    X = np.array([r[0] for r in rows], dtype=np.float64)
    Y = np.array([r[1] for r in rows], dtype=np.float64)
    return X, Y


def fit_ridge(X_train, Y_train, X_val, Y_val, on_output=None):
    """Fit ridge regression with cross-validation on lambda."""
    n, d = X_train.shape
    # Add bias column
    X_b = np.hstack([X_train, np.ones((n, 1))])
    XtX = X_b.T @ X_b
    XtY = X_b.T @ Y_train

    best_weights = None
    best_bias = None
    best_val_mse = float('inf')
    best_lambda = RIDGE_LAMBDAS[0]

    for lam in RIDGE_LAMBDAS:
        system = XtX.copy()
        system[:d, :d] += lam * np.eye(d)  # don't regularize bias

        try:
            coeffs = np.linalg.solve(system, XtY)
        except np.linalg.LinAlgError:
            continue

        W = coeffs[:d].astype(np.float32)
        b = coeffs[d].astype(np.float32)

        # Evaluate
        train_pred = X_train @ W + b
        train_mse = np.mean((train_pred - Y_train) ** 2)

        if X_val.shape[0] > 0:
            val_pred = X_val @ W + b
            val_mse = np.mean((val_pred - Y_val) ** 2)
        else:
            val_mse = train_mse

        msg = f"  ridge={lam:.4f}: train MSE={train_mse:.4f}, val MSE={val_mse:.4f}"
        if on_output:
            on_output(msg)
        else:
            print(msg)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_weights = W
            best_bias = b
            best_lambda = lam

    return best_weights, best_bias, best_lambda, best_val_mse


def export_onnx(output_path, weights, bias):
    """Export ridge regression model as ONNX: AveragePool -> Flatten -> Gemm."""
    if not HAS_ONNX:
        raise RuntimeError("onnx package not available. Cannot export model.")

    # Input: [batch, 1, 64, 64]
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 1, INPUT_H, INPUT_W])
    # Output: [batch, 3]
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, OUTPUT_COUNT])

    # AveragePool
    pool_node = helper.make_node(
        'AveragePool', inputs=['input'], outputs=['pooled'],
        kernel_shape=[POOL_SIZE, POOL_SIZE], strides=[POOL_SIZE, POOL_SIZE]
    )

    # Flatten
    flatten_node = helper.make_node(
        'Flatten', inputs=['pooled'], outputs=['flat'], axis=1
    )

    # Gemm (linear layer)
    # weights shape: [FEATURE_COUNT, OUTPUT_COUNT] -> transposed for Gemm: [OUTPUT_COUNT, FEATURE_COUNT]
    W_init = numpy_helper.from_array(weights.T.copy(), name='linear_weight')
    b_init = numpy_helper.from_array(bias.copy(), name='linear_bias')

    gemm_node = helper.make_node(
        'Gemm', inputs=['flat', 'linear_weight', 'linear_bias'], outputs=['output'],
        transB=1
    )

    graph = helper.make_graph(
        [pool_node, flatten_node, gemm_node],
        'EyebrowRidgeRegression',
        [X], [Y],
        initializer=[W_init, b_init]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.producer_name = 'VREyebrowTracker.NativeRidge'

    onnx.save(model, str(output_path))


def train_native(data_dir, train_csv, val_csv, output_path, on_output=None):
    """Full training pipeline. Returns True on success."""
    def log(msg):
        if on_output:
            on_output(msg)
        else:
            print(msg)

    log(f"Loading training data from {train_csv}...")
    X_train, Y_train = load_dataset(train_csv, data_dir)
    log(f"Loading validation data from {val_csv}...")
    X_val, Y_val = load_dataset(val_csv, data_dir)

    if X_train.shape[0] == 0:
        log("Error: No training samples found.")
        return False

    log(f"Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")
    log(f"Features: {FEATURE_COUNT} (avg pool {POOL_SIZE}x{POOL_SIZE})")
    log("Fitting ridge regression...")

    weights, bias, best_lambda, val_mse = fit_ridge(X_train, Y_train, X_val, Y_val, on_output=log)

    if weights is None:
        log("Error: Ridge regression failed.")
        return False

    log(f"Best lambda: {best_lambda}, Val MSE: {val_mse:.4f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    export_onnx(output_path, weights, bias)
    log(f"Model saved: {output_path}")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/eyebrow_images/")
    parser.add_argument("--train-csv", default="./data/train.csv")
    parser.add_argument("--val-csv", default="./data/val.csv")
    parser.add_argument("--output", default="./eyebrow_model.onnx")
    args = parser.parse_args()

    ok = train_native(args.data_dir, args.train_csv, args.val_csv, args.output)
    if ok:
        print("Done!")
    else:
        print("Failed.")
