"""Run once after `CREATE EXTENSION vector;` is available: python -m scripts.init_db"""

import asyncio

from sqlalchemy import text

from app.database import engine
from app.models import Base


async def main():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS face_embeddings_embedding_idx "
                "ON face_embeddings USING hnsw (embedding vector_cosine_ops)"
            )
        )
    print("Database siap: extension vector, tabel, dan index sudah dibuat.")


if __name__ == "__main__":
    asyncio.run(main())
