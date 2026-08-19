export interface FaceMatch {
  bbox: [number, number, number, number];
  person_id: number | null;
  name: string | null;
  similarity: number;
  live: boolean;
  liveness_score: number;
  pose: "frontal" | "left" | "right" | "unknown";
}

export interface RecognizeMessage {
  faces: FaceMatch[];
}
