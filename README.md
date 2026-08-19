# Live Face Recognition

Stack: FastAPI (Python) + InsightFace (deteksi & embedding wajah) + FaRL/`facer` (analisis wajah opsional) + PostgreSQL/pgvector, frontend React (Vite).

## 1. Database

pgvector wajib diaktifkan dulu:

```bash
sudo apt install postgresql-18-pgvector
```

Lalu, dari folder `backend/`:

```bash
source .venv/bin/activate
python -m scripts.init_db
```

Ini akan menjalankan `CREATE EXTENSION vector`, membuat tabel `persons` & `face_embeddings`, dan index HNSW untuk pencarian kemiripan.

## 2. Backend

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Endpoint:
- `POST /api/enroll` — form-data `name` + `file` (foto satu wajah) untuk mendaftarkan orang baru.
- `WS /ws/recognize` — kirim frame JPEG biner, terima JSON `{ faces: [{ bbox, person_id, name, similarity }] }`.

## 3. Frontend

```bash
cd frontend
npm run dev
```

Buka `http://localhost:5173`. Kolom kiri menampilkan live webcam dengan overlay hasil pengenalan, kolom kanan untuk mendaftarkan wajah baru.

## 4. Docker (opsional, jalankan semuanya sekaligus)

```bash
docker compose up --build
```

Ini menjalankan 3 service sekaligus: `db` (Postgres + pgvector, image `pgvector/pgvector:pg18`), `backend` (auto migrate lewat `scripts/init_db` tiap start), dan `frontend` (build statis, disajikan lewat nginx). Buka `http://localhost:5173` seperti biasa; backend tetap di `http://localhost:8000`.

- **GPU:** backend image-nya sama persis untuk CPU maupun GPU (`INSIGHTFACE_PROVIDER=auto` — cek `torch.cuda.is_available()` saat container start). Supaya GPU host benar-benar diteruskan ke container, perlu `nvidia-container-toolkit` terpasang di host, lalu jalankan dengan override tambahan:
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
  ```
  Tanpa override ini, container tetap jalan normal — cuma pakai CPU.
- Kredensial database di `docker-compose.yml` (`fastface`/`fastface`) cuma untuk container-to-container, terpisah dari `backend/.env` yang dipakai kalau jalan manual di luar Docker.
- Model InsightFace (deteksi + recognition) di-*bake* ke image backend saat build (`docker build` butuh akses internet), jadi container baru tidak perlu download apa pun saat start.
- Belum saya coba build betulan di sini (Docker tidak tersedia di environment ini) — kalau ada error `apt`/library yang kurang saat `docker compose build`, kabari saya paket apa yang diminta, biasanya cukup tambah satu baris di `backend/Dockerfile`.

## Catatan

- Threshold kecocokan wajah diatur lewat `FACE_MATCH_THRESHOLD` di `backend/.env` (default 0.5, cosine similarity). Sesuaikan berdasarkan false positive/negative yang terlihat.
- Passive liveness (anti-spoof foto/layar) aktif di `/api/enroll` dan `/ws/recognize` lewat `app/antispoof.py` (MiniFASNetV2, satu frame, tidak perlu user berkedip/gerak). Threshold-nya `LIVENESS_THRESHOLD` di `backend/.env` (default 0.7). Wajah yang gagal cek ini muncul sebagai "Spoof" dan tidak akan dicocokkan ke database.
- `INSIGHTFACE_PROVIDER` default-nya `auto` — otomatis pakai GPU NVIDIA (CUDA) kalau memang terdeteksi ada dan bisa dipakai, kalau tidak jatuh ke CPU. Bisa dipaksa manual ke `CPUExecutionProvider`/`CUDAExecutionProvider` di `.env` kalau perlu.
- `farl_engine.py` memuat model FaRL lewat `facer` secara lazy (belum dipakai di alur live recognition) — siapkan sebagai endpoint terpisah kalau butuh analisis wajah tambahan (parsing/atribut).
