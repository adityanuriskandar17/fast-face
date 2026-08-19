import asyncio

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import crud
from ..antispoof import LivenessTracker, liveness_score
from ..config import settings
from ..database import SessionLocal
from ..face_engine import detect_faces, estimate_pose

router = APIRouter(tags=["recognize"])


@router.websocket("/ws/recognize")
async def recognize(ws: WebSocket):
    await ws.accept()
    tracker = LivenessTracker()
    try:
        async with SessionLocal() as session:
            while True:
                frame_bytes = await ws.receive_bytes()
                image = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    continue

                results = []
                seen_track_ids = set()
                # detect_faces/liveness_score are CPU/GPU-bound and synchronous -- off the
                # event loop, so one slow frame on this connection doesn't stall every other
                # camera/gate connected to the same server.
                faces = await asyncio.to_thread(detect_faces, image)
                for face in faces:
                    raw_score = await asyncio.to_thread(liveness_score, image, face.bbox)
                    score, track_id = tracker.update(face.bbox, raw_score)
                    seen_track_ids.add(track_id)
                    live = score >= settings.liveness_threshold
                    person, similarity = await crud.find_closest_match(session, face.embedding)
                    matched = live and person is not None and similarity >= settings.face_match_threshold
                    results.append(
                        {
                            "bbox": face.bbox,
                            "person_id": person.id if matched else None,
                            "name": person.name if matched else ("Spoof" if not live else "Unknown"),
                            "similarity": round(similarity, 4),
                            "live": live,
                            "liveness_score": round(score, 4),
                            "pose": estimate_pose(face.kps),
                        }
                    )
                tracker.prune(seen_track_ids)
                await ws.send_json({"faces": results})
    except WebSocketDisconnect:
        pass
