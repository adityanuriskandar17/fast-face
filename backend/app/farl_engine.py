"""Optional facial analysis (parsing/attributes) built on FaRL, via the `facer` inference
library (https://github.com/FacePerceiver/facer). Kept separate from face_engine.py because
it pulls in torch and is heavier than the InsightFace detect+match path used in the live
recognition loop — call this only for on-demand analysis (e.g. an "inspect this face" endpoint).
"""

from functools import lru_cache

import numpy as np
import torch


@lru_cache
def get_face_parser():
    import facer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return facer.face_parser("farl/lapa/448", device=device), device


def parse_face_regions(image_rgb: np.ndarray, bbox: list[float]) -> dict:
    import facer

    parser, device = get_face_parser()
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    faces = {"rects": torch.tensor([bbox[:4]], device=device)}
    with torch.inference_mode():
        result = parser(image_tensor, faces)
    seg = result["seg"]["logits"].softmax(dim=1)
    labels = facer.get_label_names("farl/lapa/448")
    coverage = seg.mean(dim=(2, 3)).squeeze(0).tolist()
    return dict(zip(labels, coverage))
