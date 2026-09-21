"""Production preprocessing, reproduced so the checker stays standalone.

:func:`letterbox` is a copy of ``core.utils.ImageUtils.letterbox``. The copy
exists so the checker can be packaged without the application, and it is pinned
to the original by ``tests/test_system_check_benchmark.py``, which imports both
and asserts they produce identical arrays. If production preprocessing changes,
that test fails rather than the benchmark silently measuring something the line
no longer does.

The rest of the transform — BGR to RGB, scale to 0..1, HWC to CHW, add a batch
axis — is what Ultralytics applies internally before it hands a frame to the
ONNX session, and is reproduced here for the direct-ONNX backend.
"""

from __future__ import annotations

import cv2
import numpy as np


def letterbox(
    img: np.ndarray,
    size: tuple[int, int] = (640, 640),
    fill_color: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Resize an image to ``size`` while keeping aspect ratio via padding.

    Args:
        img: Source image as HxWx3 BGR.
        size: Target ``(height, width)``.
        fill_color: Padding colour.

    Returns:
        A ``size``-shaped BGR image.
    """
    h, w = img.shape[:2]
    target_h, target_w = size
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded_img: np.ndarray = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    padded_img[top : top + new_h, left : left + new_w] = resized_img
    if padded_img.shape[:2] != (target_h, target_w):
        padded_img = cv2.resize(
            padded_img, (target_w, target_h), interpolation=cv2.INTER_AREA
        )
    return padded_img


def to_model_input(letterboxed_bgr: np.ndarray) -> np.ndarray:
    """Convert a letterboxed BGR frame into the ONNX input tensor.

    Args:
        letterboxed_bgr: HxWx3 BGR image already at the model's input size.

    Returns:
        A ``(1, 3, H, W)`` float32 array scaled to 0..1 in RGB order.
    """
    rgb = cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    batched: np.ndarray = np.ascontiguousarray(tensor[np.newaxis, ...])
    return batched


def synthetic_frame(width: int, height: int, seed: int = 20260921) -> np.ndarray:
    """Build a deterministic BGR frame at the camera's configured resolution.

    Benchmarking against a fixed synthetic frame keeps two machines comparable
    without shipping inspection images between sites, and keeps the checker
    away from production evidence entirely. The frame is structured rather than
    pure noise — noise compresses and resizes differently from real content, and
    the resize is a measurable part of the preprocessing cost this benchmark
    reports.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        seed: Fixed seed, so every machine letterboxes identical pixels.

    Returns:
        An HxWx3 uint8 BGR image.
    """
    rng = np.random.default_rng(seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # A smooth background gradient: cheap to build, and unlike noise it
    # survives INTER_AREA downscaling the way real imagery does.
    gradient = np.linspace(20, 200, width, dtype=np.float32)
    frame[:, :, 0] = gradient.astype(np.uint8)
    frame[:, :, 1] = np.linspace(200, 20, width, dtype=np.float32).astype(np.uint8)
    frame[:, :, 2] = np.linspace(60, 180, height, dtype=np.float32)[:, None].astype(np.uint8)

    # A row of saturated blocks standing in for the coloured cable ends this
    # station inspects, so postprocessing has candidate regions to score.
    block_count = 6
    block_w = max(8, width // (block_count * 3))
    block_h = max(8, height // 6)
    top = height // 2 - block_h // 2
    palette = (
        (0, 0, 220),
        (0, 200, 0),
        (0, 140, 255),
        (0, 230, 230),
        (30, 30, 30),
        (30, 30, 30),
    )
    for index, colour in enumerate(palette):
        left = int((index + 0.5) * width / block_count) - block_w // 2
        left = max(0, min(width - block_w, left))
        frame[top : top + block_h, left : left + block_w] = colour

    # A light speckle so the encoder-like paths are not fed a constant image.
    noise = rng.integers(0, 12, size=(height, width, 1), dtype=np.uint8)
    speckled: np.ndarray = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return speckled
