import asyncio

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from .. import crud
from ..antispoof import liveness_score
from ..config import settings
from ..database import get_session
from ..face_engine import detect_faces
from ..schemas import EnrollResponse

router = APIRouter(prefix="/api", tags=["enroll"])


async def _decode_single_live_face(file: UploadFile) -> list[float]:
    """Decode `file` and return its embedding, after checking it holds exactly one live face."""
    raw = await file.read()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, f"Foto '{file.filename}' bukan gambar yang valid")

    faces = await asyncio.to_thread(detect_faces, image)
    if not faces:
        raise HTTPException(422, f"Tidak ada wajah terdeteksi pada foto '{file.filename}'")
    if len(faces) > 1:
        raise HTTPException(422, f"Lebih dari satu wajah terdeteksi pada foto '{file.filename}'")

    score = await asyncio.to_thread(liveness_score, image, faces[0].bbox)
    if score < settings.liveness_threshold:
        raise HTTPException(422, f"Foto '{file.filename}' terdeteksi sebagai foto/layar, gunakan wajah asli langsung dari kamera")

    return faces[0].embedding


@router.post("/enroll", response_model=EnrollResponse)
async def enroll(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    merge_into: int | None = Form(None),
    force_new: bool = Form(False),
    session: AsyncSession = Depends(get_session),
):
    if not files:
        raise HTTPException(422, "Butuh minimal satu foto")

    # Multiple angles (straight/left/right) give the matcher more to compare against, since
    # find_closest_match searches across every embedding a person has, not just one.
    embeddings = [await _decode_single_live_face(f) for f in files]

    if merge_into is not None:
        person = await crud.get_person(session, merge_into)
        if person is None:
            raise HTTPException(404, "Anggota tujuan penggabungan tidak ditemukan")
        for embedding in embeddings:
            await crud.add_embedding_to_person(session, person.id, embedding)
        return EnrollResponse(person_id=person.id, name=person.name)

    if not force_new:
        # Same face already enrolled under someone else? Flag it instead of silently creating
        # a duplicate identity -- staff decides whether to merge or register anyway.
        best_person, best_similarity = None, 0.0
        for embedding in embeddings:
            match, similarity = await crud.find_closest_match(session, embedding)
            if match is not None and similarity > best_similarity:
                best_person, best_similarity = match, similarity
        if best_person is not None and best_similarity >= settings.face_match_threshold:
            raise HTTPException(
                409,
                {
                    "duplicate_person_id": best_person.id,
                    "duplicate_name": best_person.name,
                    "similarity": round(best_similarity, 4),
                },
            )

    person = await crud.create_person_with_embedding(session, name, embeddings[0])
    for embedding in embeddings[1:]:
        await crud.add_embedding_to_person(session, person.id, embedding)

    return EnrollResponse(person_id=person.id, name=person.name)
