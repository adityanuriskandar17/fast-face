from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import FaceEmbedding, Person


async def create_person_with_embedding(session: AsyncSession, name: str, embedding: list[float]) -> Person:
    person = Person(name=name)
    person.embeddings.append(FaceEmbedding(embedding=embedding))
    session.add(person)
    await session.commit()
    await session.refresh(person)
    return person


async def add_embedding_to_person(session: AsyncSession, person_id: int, embedding: list[float]) -> None:
    session.add(FaceEmbedding(person_id=person_id, embedding=embedding))
    await session.commit()


async def get_person(session: AsyncSession, person_id: int) -> Person | None:
    return await session.get(Person, person_id)


async def find_closest_match(session: AsyncSession, embedding: list[float]) -> tuple[Person | None, float]:
    """Cosine distance via pgvector's `<=>` operator; similarity = 1 - distance."""
    distance = FaceEmbedding.embedding.cosine_distance(embedding)
    stmt = (
        select(Person, distance.label("distance"))
        .join(FaceEmbedding, FaceEmbedding.person_id == Person.id)
        .order_by(distance)
        .limit(1)
    )
    row = (await session.execute(stmt)).first()
    if row is None:
        return None, 0.0
    person, dist = row
    return person, 1.0 - dist
