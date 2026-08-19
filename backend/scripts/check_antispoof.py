"""Smoke check for app.antispoof: run `python -m scripts.check_antispoof` from backend/.

Uses insightface's own bundled sample photo (real faces) as a control: a correct
pipeline must score these clearly above the 0.5 midpoint, not just "in range".
"""

import cv2
import insightface

from app.antispoof import LivenessTracker, liveness_score
from app.face_engine import detect_faces

sample = insightface.__path__[0] + "/data/images/t1.jpg"
image = cv2.imread(sample)
faces = detect_faces(image)
assert faces, f"no faces detected in control image {sample}"

scores = [liveness_score(image, f.bbox) for f in faces]
assert all(0.0 <= s <= 1.0 for s in scores), f"score out of range: {scores}"
assert all(s > 0.5 for s in scores), f"real faces scored as spoof, preprocessing likely broken: {scores}"
print(f"ok: {len(scores)} real faces, scores={[round(s, 3) for s in scores]}")

# LivenessTracker: one bad-angle frame (0.2) among good ones must not flip the smoothed verdict.
tracker = LivenessTracker(window=5)
bbox = [100.0, 100.0, 200.0, 220.0]
for raw in [0.9, 0.9, 0.9, 0.2, 0.9]:
    smoothed, track_id_a = tracker.update(bbox, raw)
assert smoothed > 0.7, f"single bad frame flipped the smoothed score: {smoothed}"

far_bbox = [500.0, 400.0, 600.0, 520.0]
_, track_id_b = tracker.update(far_bbox, 0.9)
assert track_id_b != track_id_a, "a far-away face was merged into the same track"

tracker.prune({track_id_b})  # drop track_id_a as if that face left the frame
smoothed_again, track_id_a2 = tracker.update(bbox, 0.9)
assert track_id_a2 != track_id_a, "pruned track was not actually dropped"
print(f"ok: tracker smoothed={smoothed:.3f}, correctly separated and pruned tracks")
