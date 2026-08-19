FROM python:3.11-slim

WORKDIR /app

# libglib2.0-0: opencv-python-headless still dlopens it. libgomp1: OpenMP runtime torch/onnx need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# Bake in the 2 InsightFace models this app actually loads (detection + recognition --
# see allowed_modules in face_engine.py) so a fresh container isn't stuck downloading them
# from the internet on its first real request. No GPU is expected at build time; that's
# fine, this step only needs to fetch and cache the .onnx files.
RUN python -c "from app.face_engine import get_face_app; get_face_app()"

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
