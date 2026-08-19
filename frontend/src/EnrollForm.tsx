import { useEffect, useRef, useState } from "react";
import LiveFace, { type LiveFaceHandle } from "./LiveFace";
import type { FaceMatch } from "./types";

const API_URL = "http://localhost:8000/api/enroll";
const POSES = ["Lihat lurus ke kamera", "Putar sedikit ke kiri", "Putar sedikit ke kanan"];
const POSE_KEYS: FaceMatch["pose"][] = ["frontal", "left", "right"];
const STABLE_FRAMES_NEEDED = 2; // consecutive matching frames (~600ms @ 300ms/frame) before auto-capture fires

type Tone = "idle" | "busy" | "success" | "error";
type Shot = { file: File; previewUrl: string };
type Duplicate = { personId: number; name: string; similarity: number };

export default function EnrollForm() {
  const liveFaceRef = useRef<LiveFaceHandle>(null);
  const [liveFaces, setLiveFaces] = useState<FaceMatch[]>([]);
  const [name, setName] = useState("");
  const [shots, setShots] = useState<Shot[]>([]);
  const [message, setMessage] = useState("");
  const [tone, setTone] = useState<Tone>("idle");
  const [duplicate, setDuplicate] = useState<Duplicate | null>(null);
  const capturingRef = useRef(false);
  const stableFramesRef = useRef(0);

  function resetShots() {
    shots.forEach((s) => URL.revokeObjectURL(s.previewUrl));
    setShots([]);
    setDuplicate(null);
    stableFramesRef.current = 0;
  }

  function addShots(files: File[]) {
    setShots((prev) => [...prev, ...files.map((file) => ({ file, previewUrl: URL.createObjectURL(file) }))]);
  }

  async function handleCapture() {
    if (capturingRef.current) return;
    capturingRef.current = true;
    try {
      setTone("busy");
      setMessage(`Mengambil pose ${shots.length + 1}/${POSES.length}…`);
      const blob = (await liveFaceRef.current?.captureFrame()) ?? null;
      if (!blob) {
        setTone("error");
        setMessage("Kamera belum siap — coba lagi sebentar.");
        return;
      }
      addShots([new File([blob], `pose-${shots.length + 1}.jpg`, { type: blob.type })]);
      setTone("idle");
      const remaining = POSES.length - (shots.length + 1);
      setMessage(remaining > 0 ? `Pose ${shots.length + 1} diambil. Lanjut: ${POSES[shots.length + 1]}.` : "3 pose siap. Klik Daftarkan Wajah.");
    } finally {
      stableFramesRef.current = 0;
      capturingRef.current = false;
    }
  }

  // Auto-capture: once the live camera shows exactly one real (non-spoof) face holding the
  // pose this step wants, for a couple of frames in a row (so a mid-turn glance doesn't
  // trigger it), fire the same capture the manual button does.
  useEffect(() => {
    if (shots.length >= POSES.length) return;
    const targetPose = POSE_KEYS[shots.length];
    const onlyFace = liveFaces.length === 1 ? liveFaces[0] : null;
    const isMatch = !!onlyFace && onlyFace.live && onlyFace.pose === targetPose;
    stableFramesRef.current = isMatch ? stableFramesRef.current + 1 : 0;
    if (stableFramesRef.current >= STABLE_FRAMES_NEEDED) {
      handleCapture();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveFaces]);

  async function submitEnroll(extra: { merge_into?: number; force_new?: boolean }) {
    const form = new FormData();
    form.append("name", name);
    shots.forEach((s) => form.append("files", s.file));
    if (extra.merge_into !== undefined) form.append("merge_into", String(extra.merge_into));
    if (extra.force_new) form.append("force_new", "true");

    setTone("busy");
    setDuplicate(null);
    setMessage(extra.merge_into !== undefined ? "Menambahkan foto ke anggota…" : "Menyimpan ke roster…");
    const res = await fetch(API_URL, { method: "POST", body: form });
    const data = await res.json();
    if (res.ok) {
      setTone("success");
      setMessage(`Terdaftar: ${data.name} · ID ${data.person_id}`);
      setName("");
      resetShots();
    } else if (res.status === 409) {
      setTone("idle");
      setMessage("");
      setDuplicate({ personId: data.detail.duplicate_person_id, name: data.detail.duplicate_name, similarity: data.detail.similarity });
    } else {
      setTone("error");
      setMessage(`Gagal mendaftarkan: ${data.detail}`);
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!shots.length || !name) return;
    submitEnroll({});
  }

  const nextPose = POSES[shots.length];
  const hint = liveHint(liveFaces, POSE_KEYS[shots.length]);

  return (
    <>
      <section>
        <h2 className="panel__title">Kamera pendaftaran</h2>
        <LiveFace ref={liveFaceRef} onFaces={setLiveFaces} />
      </section>

      <section className="roster">
        <span className="eyebrow">Roster</span>
        <h2 className="panel__title">Daftarkan wajah baru</h2>
        <form onSubmit={handleSubmit}>
          <div className="field">
            <label htmlFor="member-name">Nama anggota</label>
            <input
              id="member-name"
              type="text"
              placeholder="Nama lengkap"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>

          <div className="roster__actions">
            {nextPose ? (
              <>
                <button type="button" className="btn btn--ghost" onClick={handleCapture}>
                  Ambil pose {shots.length + 1}/{POSES.length}: {nextPose}
                </button>
                <p className="roster__hint">{hint}</p>
              </>
            ) : (
              <p className="roster__message">3 pose sudah diambil.</p>
            )}
            {shots.length > 0 && (
              <button type="button" className="btn btn--ghost" onClick={resetShots}>
                Ulangi dari awal
              </button>
            )}

            <div className="divider-word">atau</div>

            <label className="btn btn--ghost btn--file">
              Pilih foto dari perangkat (bisa lebih dari satu)
              <input
                className="file-input"
                type="file"
                accept="image/*"
                multiple
                onChange={(e) => e.target.files && addShots(Array.from(e.target.files))}
              />
            </label>
          </div>

          {shots.length > 0 && (
            <div className="preview">
              {shots.map((s, i) => (
                <img key={s.previewUrl} src={s.previewUrl} alt={`Pose ${i + 1}`} />
              ))}
            </div>
          )}

          {duplicate ? (
            <div className="roster__duplicate">
              <p className="roster__message">
                Wajah ini {(duplicate.similarity * 100).toFixed(0)}% mirip anggota yang sudah terdaftar:{" "}
                <strong>{duplicate.name}</strong>.
              </p>
              <div className="roster__actions">
                <button type="button" className="btn btn--ghost" onClick={() => submitEnroll({ merge_into: duplicate.personId })}>
                  Orang yang sama — tambahkan foto ini ke {duplicate.name}
                </button>
                <button type="button" className="btn btn--ghost" onClick={() => submitEnroll({ force_new: true })}>
                  Bukan, ini anggota baru — daftarkan terpisah
                </button>
                <button type="button" className="btn btn--ghost" onClick={() => setDuplicate(null)}>
                  Batal, periksa lagi
                </button>
              </div>
            </div>
          ) : (
            <button type="submit" className="btn btn--primary" disabled={!shots.length || !name}>
              Daftarkan Wajah
            </button>
          )}

          {message && <p className={`roster__message roster__message--${tone}`}>{message}</p>}
        </form>
      </section>
    </>
  );
}

function liveHint(liveFaces: FaceMatch[], targetPose: FaceMatch["pose"]): string {
  if (liveFaces.length === 0) return "Menunggu wajah terdeteksi…";
  if (liveFaces.length > 1) return "Hanya boleh 1 wajah di kamera";
  const face = liveFaces[0];
  if (!face.live) return "Tidak bisa memverifikasi wajah — coba dekatkan ke kamera";
  if (face.pose === targetPose) return "Pas! Menahan posisi…";
  return "Sesuaikan arah wajah sesuai instruksi di atas";
}
