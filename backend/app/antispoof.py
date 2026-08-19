"""Passive liveness check: single-frame real-vs-spoof (photo/screen) classifier.

Network and preprocessing convention adapted from minivision-ai/Silent-Face-Anti-Spoofing
(Apache-2.0): https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
Weights: resources/anti_spoof_models/{2.7_80x80_MiniFASNetV2,4_0_0_80x80_MiniFASNetV1SE}.pth
from the same repo -- the same 2-model ensemble upstream uses, each with its own crop
scale (2.7x / 4.0x context window around the face bbox); final score is their average.
"""

from collections import OrderedDict, deque
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Linear,
    Module,
    PReLU,
    ReLU,
    Sequential,
    Sigmoid,
)

WEIGHTS_DIR = Path(__file__).parent / "weights"
INPUT_SIZE = 80
REAL_CLASS_INDEX = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_KEEP_V2 = [
    32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13,
    13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
    52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26,
    128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512,
]
_KEEP_V1 = [
    32, 32, 103, 103, 64, 13, 13, 64, 26, 26, 64, 13,
    13, 64, 52, 52, 64, 231, 231, 128, 154, 154, 128, 52,
    52, 128, 26, 26, 128, 52, 52, 128, 26, 26, 128, 26, 26,
    128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512,
]

# (weights file, keep-channel table, use squeeze-excite blocks, crop scale around bbox)
_MODEL_SPECS = [
    ("2.7_80x80_MiniFASNetV2.pth", _KEEP_V2, False, 2.7),
    ("4_0_0_80x80_MiniFASNetV1SE.pth", _KEEP_V1, True, 4.0),
]


class _ConvBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class _LinearBlock(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class _DepthWise(Module):
    def __init__(self, c1, c2, c3, residual, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super().__init__()
        self.conv = _ConvBlock(c1[0], c1[1], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_dw = _ConvBlock(c2[0], c2[1], groups=c2[0], kernel=kernel, stride=stride, padding=padding)
        self.project = _LinearBlock(c3[0], c3[1], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.residual = residual

    def forward(self, x):
        out = self.project(self.conv_dw(self.conv(x)))
        return x + out if self.residual else out


class _SEModule(Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.bn1 = BatchNorm2d(channels // reduction)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.bn2 = BatchNorm2d(channels)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        s = self.relu(self.bn1(self.fc1(self.avg_pool(x))))
        s = self.sigmoid(self.bn2(self.fc2(s)))
        return x * s


class _DepthWiseSE(Module):
    """Like `_DepthWise(residual=True)`, plus a squeeze-excite gate before the skip-add."""

    def __init__(self, c1, c2, c3, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, se_reduct=4):
        super().__init__()
        self.conv = _ConvBlock(c1[0], c1[1], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_dw = _ConvBlock(c2[0], c2[1], groups=c2[0], kernel=kernel, stride=stride, padding=padding)
        self.project = _LinearBlock(c3[0], c3[1], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.se_module = _SEModule(c3[1], se_reduct)

    def forward(self, x):
        out = self.se_module(self.project(self.conv_dw(self.conv(x))))
        return x + out


class _Residual(Module):
    """`num_block` residual depth-wise blocks; the last one gets an SE gate when `se=True`."""

    def __init__(self, c1, c2, c3, num_block, groups, se=False, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        blocks = []
        for i in range(num_block):
            if se and i == num_block - 1:
                blocks.append(_DepthWiseSE(c1[i], c2[i], c3[i], kernel=kernel, stride=stride, padding=padding, groups=groups))
            else:
                blocks.append(_DepthWise(c1[i], c2[i], c3[i], residual=True, kernel=kernel, stride=stride, padding=padding, groups=groups))
        self.model = Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class MiniFASNet(Module):
    """(80x80) flops: 0.044, params: ~0.43 -- see module docstring for the source repo."""

    def __init__(self, keep, se=False, embedding_size=128, conv6_kernel=(5, 5), num_classes=3):
        super().__init__()
        self.conv1 = _ConvBlock(3, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = _ConvBlock(keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1])
        self.conv_23 = _DepthWise(
            (keep[1], keep[2]), (keep[2], keep[3]), (keep[3], keep[4]),
            residual=False, stride=(2, 2), groups=keep[3],
        )
        self.conv_3 = _Residual(
            [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])],
            [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])],
            [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])],
            num_block=4, groups=keep[4], se=se,
        )
        self.conv_34 = _DepthWise(
            (keep[16], keep[17]), (keep[17], keep[18]), (keep[18], keep[19]),
            residual=False, stride=(2, 2), groups=keep[19],
        )
        self.conv_4 = _Residual(
            [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), (keep[28], keep[29]), (keep[31], keep[32]), (keep[34], keep[35])],
            [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), (keep[29], keep[30]), (keep[32], keep[33]), (keep[35], keep[36])],
            [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), (keep[30], keep[31]), (keep[33], keep[34]), (keep[36], keep[37])],
            num_block=6, groups=keep[19], se=se,
        )
        self.conv_45 = _DepthWise(
            (keep[37], keep[38]), (keep[38], keep[39]), (keep[39], keep[40]),
            residual=False, stride=(2, 2), groups=keep[40],
        )
        self.conv_5 = _Residual(
            [(keep[40], keep[41]), (keep[43], keep[44])],
            [(keep[41], keep[42]), (keep[44], keep[45])],
            [(keep[42], keep[43]), (keep[45], keep[46])],
            num_block=2, groups=keep[40], se=se,
        )
        self.conv_6_sep = _ConvBlock(keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = _LinearBlock(keep[47], keep[48], groups=keep[48], kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.drop = Dropout(p=0.2)
        self.prob = Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        return self.prob(out)


def _crop_for_model(image_bgr: np.ndarray, bbox: list[float], crop_scale: float) -> np.ndarray:
    """Expand `bbox` (x1,y1,x2,y2) into the square context window the model was trained on."""
    src_h, src_w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w, box_h = x2 - x1, y2 - y1

    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, crop_scale)
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = x1 + box_w / 2, y1 + box_h / 2

    left, top = cx - new_w / 2, cy - new_h / 2
    right, bottom = cx + new_w / 2, cy + new_h / 2
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > src_w - 1:
        left -= right - src_w + 1
        right = src_w - 1
    if bottom > src_h - 1:
        top -= bottom - src_h + 1
        bottom = src_h - 1

    crop = image_bgr[int(top) : int(bottom) + 1, int(left) : int(right) + 1]
    return cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))


@lru_cache
def _get_models() -> list[tuple[Module, float]]:
    models = []
    for filename, keep, se, crop_scale in _MODEL_SPECS:
        model = MiniFASNet(keep, se=se)
        state_dict = torch.load(WEIGHTS_DIR / filename, map_location="cpu")
        stripped = OrderedDict((k[len("module.") :] if k.startswith("module.") else k, v) for k, v in state_dict.items())
        model.load_state_dict(stripped)
        model.eval()
        models.append((model.to(DEVICE), crop_scale))
    return models


def liveness_score(image_bgr: np.ndarray, bbox: list[float]) -> float:
    """Probability in [0, 1] that the face at `bbox` (x1,y1,x2,y2) is a live person, not a photo/screen.

    Averages the 2-model ensemble's softmax outputs (each model has its own crop scale).
    """
    real_probs = []
    with torch.no_grad():
        for model, crop_scale in _get_models():
            patch = _crop_for_model(image_bgr, bbox, crop_scale)
            # Matches upstream: their to_tensor's /255 division is commented out ("backward
            # compatibility"), so the checkpoints were trained on raw [0,255] pixel values.
            tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
            probs = torch.softmax(model(tensor), dim=1)
            real_probs.append(probs[0, REAL_CLASS_INDEX].item())
    return sum(real_probs) / len(real_probs)


class LivenessTracker:
    """Smooths `liveness_score` across frames so one bad-angle frame doesn't flip the verdict.

    Faces are matched frame-to-frame by bbox-center proximity (no identity assumed --
    smoothing has to happen *before* the match/spoof decision exists). Call `update` once
    per detected face each frame, then `prune` with the track ids seen that frame so tracks
    for faces that left the frame don't linger and confuse the next arrival.
    """

    def __init__(self, window: int = 5, match_dist: float = 100.0):
        self.window = window
        self.match_dist = match_dist
        self._scores: dict[int, deque] = {}
        self._centers: dict[int, tuple[float, float]] = {}
        self._next_id = 0

    def update(self, bbox: list[float], score: float) -> tuple[float, int]:
        """Feed this frame's raw score for `bbox`; return (smoothed_score, track_id)."""
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        track_id = self._closest_track(cx, cy)
        if track_id is None:
            track_id = self._next_id
            self._next_id += 1
            self._scores[track_id] = deque(maxlen=self.window)
        self._scores[track_id].append(score)
        self._centers[track_id] = (cx, cy)
        recent = self._scores[track_id]
        return sum(recent) / len(recent), track_id

    def _closest_track(self, cx: float, cy: float) -> int | None:
        best_id, best_dist = None, self.match_dist
        for track_id, (tx, ty) in self._centers.items():
            dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
            if dist < best_dist:
                best_id, best_dist = track_id, dist
        return best_id

    def prune(self, seen_track_ids: set) -> None:
        for track_id in list(self._scores):
            if track_id not in seen_track_ids:
                del self._scores[track_id]
                del self._centers[track_id]
