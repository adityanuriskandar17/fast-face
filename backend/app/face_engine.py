import ctypes
import glob
import os
import site
from functools import lru_cache

import numpy as np
import torch

# onnxruntime-gpu's CUDA provider needs cuBLAS/cuDNN .so files. `torch` already vendors
# matching ones under site-packages/nvidia/*/lib as pip deps, but onnxruntime's dlopen
# won't find them unless already loaded into the process -- and LD_LIBRARY_PATH set from
# Python has no effect on dlopen (glibc reads it once at process start). Preloading the
# files directly (before onnxruntime ever dlopens its CUDA provider) sidesteps both.
for _site_dir in site.getsitepackages():
    for _so in glob.glob(os.path.join(_site_dir, "nvidia", "*", "lib", "*.so*")):
        try:
            ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

from insightface.app import FaceAnalysis  # noqa: E402

from .config import settings


class DetectedFace:
    def __init__(self, bbox: np.ndarray, embedding: np.ndarray, kps: np.ndarray):
        self.bbox = bbox.tolist()
        self.embedding = embedding.tolist()
        self.kps = kps.tolist()


def _resolve_provider() -> str:
    """"auto" picks CUDA only if a GPU is actually usable -- same image/env works unchanged
    on a GPU box or a CPU-only one. An explicit setting (e.g. forcing CPU for a comparison)
    still wins over the probe."""
    if settings.insightface_provider != "auto":
        return settings.insightface_provider
    return "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"


@lru_cache
def get_face_app() -> FaceAnalysis:
    # Only bbox + embedding are used (see DetectedFace below) -- buffalo_l's age/gender and
    # 3D/2D landmark models would otherwise run on every face for no reason.
    provider = _resolve_provider()
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=[provider],
    )
    det_size = (settings.insightface_det_size, settings.insightface_det_size)
    app.prepare(ctx_id=0 if provider == "CUDAExecutionProvider" else -1, det_size=det_size)
    return app


def detect_faces(image_bgr: np.ndarray) -> list[DetectedFace]:
    faces = get_face_app().get(image_bgr)
    return [DetectedFace(bbox=f.bbox, embedding=f.normed_embedding, kps=f.kps) for f in faces]


def estimate_pose(kps: list[list[float]]) -> str:
    """Rough yaw bucket ("frontal"/"left"/"right") from the 5-point landmarks SCRFD already
    gives us for free (order: eye, eye, nose, mouth, mouth -- by on-screen x position, not
    anatomy). `image_bgr` here is the raw, unmirrored capture (see LiveFace.tsx, which only
    mirrors the on-screen <video> for display) -- so turning the head to *the user's own*
    left swings the nose towards the eye with the larger x, i.e. ratio -> 1.
    """
    (x0, _), (x1, _), (nx, _) = kps[0], kps[1], kps[2]
    lo, hi = min(x0, x1), max(x0, x1)
    span = hi - lo
    if span <= 1:
        return "unknown"
    ratio = (nx - lo) / span
    if ratio > 0.62:
        return "left"
    if ratio < 0.38:
        return "right"
    if 0.42 <= ratio <= 0.58:
        return "frontal"
    return "unknown"
