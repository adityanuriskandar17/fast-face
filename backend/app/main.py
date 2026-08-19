from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import enroll, recognize

app = FastAPI(title="Live Face Recognition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(enroll.router)
app.include_router(recognize.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
