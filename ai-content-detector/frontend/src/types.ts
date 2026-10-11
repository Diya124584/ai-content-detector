export type Prediction = "AI-generated" | "Human-written" | "Uncertain";

export interface TrustScore {
  value: number; // 0-100, a reliability estimate; NOT the AI probability
  status: string;
  components: Record<string, number | null>;
  missing_components: string[];
  ood_flags: string[];
  disclaimer: string;
}

export interface ExplanationItem {
  factor: string;
  direction: "toward_ai" | "toward_human" | "neutral" | "caveat";
  strength: number;
  contribution: number;
  detail: string;
}

export interface DetectResponse {
  prediction: Prediction;
  ai_probability: number;
  human_probability: number;
  detection_score: number;
  trust_score: TrustScore;
  signals: Record<string, number | null>;
  signal_scores: Record<string, number | null>;
  weights_used: Record<string, number>;
  explanation: ExplanationItem[];
  warnings: string[];
  features: Record<string, number>;
  components_active: string[];
  model_version: string;
  disclaimer: string;
  history_id: number | null;
}

export interface HistoryRecord {
  id: number;
  text_hash: string;
  prediction: Prediction;
  ai_probability: number;
  human_probability: number;
  detection_score: number;
  trust_score: number | null;
  model_version: string;
  created_at: string;
}

export interface HistoryList {
  total: number;
  items: HistoryRecord[];
}
