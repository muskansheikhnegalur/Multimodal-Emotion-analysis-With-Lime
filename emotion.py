# =============================================================================
# MULTIMODAL EMOTION-AWARE AI ASSISTANT
# M.Tech Research Project — IEEE-Level Production Build
# Fixes: fusion explanation duplication, LIME narrative bug, VA fallback,
#        function ordering, section bleed, DB preview leak, widget key guard
# =============================================================================

import streamlit as st
import numpy as np
import os
import datetime
import pandas as pd
import plotly.express as px
import tempfile
import sqlite3
import sounddevice as sd
import wavio
from collections import deque
import torch
import librosa
import whisper
import warnings
import gc
import logging
from lime.lime_text import LimeTextExplainer

from transformers import (
    pipeline,
    Wav2Vec2FeatureExtractor,
    AutoModelForAudioClassification,
)

# =============================================================================
# SECTION 1 — CONFIGURATION & LOGGING
# =============================================================================

logging.basicConfig(
    filename="emotion_app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

sd.default.samplerate = 16000
sd.default.channels = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DB_PATH = os.path.join(BASE_DIR, "emotion_db.sqlite")

if "reset_flag" not in st.session_state:
    st.session_state.reset_flag = False

# =============================================================================
# SECTION 2 — UNIFIED EMOTION ONTOLOGY
# =============================================================================

CANONICAL_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "neutral", "relaxed"]

EMOTION_LABEL_MAP = {
    "happy": "joy", "happiness": "joy", "joyful": "joy", "excited": "joy",
    "joy": "joy",
    "sad": "sadness", "sadness": "sadness", "depressed": "sadness",
    "angry": "anger", "anger": "anger", "frustration": "anger", "disgust": "anger",
    "fearful": "fear", "fear": "fear", "anxious": "fear",
    "surprised": "surprise", "surprise": "surprise",
    "relaxed": "relaxed", "calm": "relaxed",
    "neutral": "neutral",
}

EMOTION_COLORS = {
    "joy": "#FFD700",
    "sadness": "#4169E1",
    "anger": "#DC143C",
    "fear": "#800080",
    "surprise": "#FF8C00",
    "relaxed": "#3CB371",
    "neutral": "#808080",
}

# =============================================================================
# SECTION 3 — UTILITY FUNCTIONS
# =============================================================================

def normalize_emotion(label: str) -> str:
    """Canonical emotion normalization — single source of truth."""
    if not label:
        return "neutral"
    return EMOTION_LABEL_MAP.get(str(label).lower().strip(), "neutral")


def safe_float(x, default: float = 0.0) -> float:
    """Safe numeric conversion — never raises."""
    try:
        val = float(x)
        return val if np.isfinite(val) else default
    except Exception:
        return default


def calculate_intensity(valence: float, arousal: float) -> float:
    v = safe_float(valence, 0.5)
    a = safe_float(arousal, 0.5)
    return round(float(a * 0.6 + abs(v - 0.5) * 0.4), 4)


def va_to_emotion(valence: float, arousal: float) -> str:
    if valence > 0.55 and arousal > 0.55:
        return "joy"
    elif valence < 0.45 and arousal > 0.55:
        return "anger"
    elif valence < 0.45 and arousal < 0.45:
        return "sadness"
    elif arousal > 0.7:
        return "surprise"
    elif valence > 0.55:
        return "relaxed"
    return "neutral"


def get_emotion_va_defaults(emotion: str):
    """Returns (valence, arousal, intensity) for an emotion — used only as fallback."""
    mapping = {
        "joy":      (0.85, 0.75, 0.80),
        "sadness":  (0.20, 0.30, 0.35),
        "anger":    (0.20, 0.85, 0.75),
        "fear":     (0.15, 0.80, 0.70),
        "surprise": (0.60, 0.80, 0.68),
        "relaxed":  (0.75, 0.25, 0.30),
        "neutral":  (0.50, 0.50, 0.30),
    }
    return mapping.get(emotion, (0.50, 0.50, 0.30))

# =============================================================================
# SECTION 4 — DATABASE
# =============================================================================

@st.cache_resource(show_spinner=False)
def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS emotions (
            timestamp         TEXT,
            emotion           TEXT,
            fatigue           TEXT,
            empathy           TEXT,
            intensity         REAL,
            message           TEXT,
            cbt_prompt        TEXT,
            valence           REAL,
            arousal           REAL,
            text_confidence   REAL,
            audio_confidence  REAL,
            fusion_confidence REAL,
            shap_words        TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_emotion_record(values: tuple):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            INSERT INTO emotions
            (timestamp, emotion, fatigue, empathy, intensity, message,
             cbt_prompt, valence, arousal, text_confidence,
             audio_confidence, fusion_confidence, shap_words)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, values)
        conn.commit()
    except Exception as e:
        logging.error(f"DB insert error: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def load_db_records(limit: int = 200) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
        df = pd.read_sql_query(
            f"SELECT * FROM emotions ORDER BY timestamp DESC LIMIT {limit}", conn
        )
        return df
    except Exception as e:
        logging.error(f"DB load error: {e}")
        return pd.DataFrame()
    finally:
        try:
            conn.close()
        except Exception:
            pass

# =============================================================================
# SECTION 5 — MODEL LOADING (cached, safe)
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    try:
        return whisper.load_model("tiny")
    except Exception as e:
        logging.error(f"Whisper load error: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_text_model():
    try:
        device_id = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
            device=device_id,
        )
        return pipe
    except Exception as e:
        logging.error(f"Text model load error: {e}")
        try:
            return pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True,
                device=-1,
            )
        except Exception:
            return None


@st.cache_resource(show_spinner=False)
def load_audio_model():
    MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    try:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
        model.config.id2label = {
            int(i): str(v).lower()
            for i, v in model.config.id2label.items()
        }
        model = model.to(DEVICE)
        return processor, model
    except Exception as e:
        logging.error(f"Audio model load error: {e}")
        return None, None


@st.cache_resource(show_spinner=False)
def load_lime_explainer():
    """Initialize LIME TextExplainer — no model files required."""
    try:
        explainer = LimeTextExplainer(
            class_names=CANONICAL_EMOTIONS,
            random_state=42,
        )
        logging.info("LIME explainer initialized successfully")
        return explainer
    except Exception as e:
        logging.error(f"LIME load error: {e}")
        return None

# =============================================================================
# SECTION 6 — MODEL WARMUP
# =============================================================================

def warmup_models(text_pipe, audio_proc, audio_mdl):
    try:
        if text_pipe is not None:
            text_pipe("hello world")
        if audio_mdl is not None and audio_proc is not None:
            dummy = np.zeros(16000, dtype=np.float32)
            inputs = audio_proc(dummy, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                _ = audio_mdl(**inputs)
        logging.info("Model warmup complete")
    except Exception as e:
        logging.error(f"Warmup error: {e}")

# =============================================================================
# SECTION 7 — AUDIO UTILITIES
# =============================================================================

def extract_prosody_features(wav_path: str) -> dict:
    """Returns pitch, tempo, energy, duration from audio file."""
    defaults = {"mean_pitch": 150.0, "tempo": 100.0, "mean_energy": 0.01, "duration": 5.0}
    try:
        y, sr = librosa.load(wav_path, sr=16000)
        if len(y) < 1000:
            return defaults

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = [
            pitches[magnitudes[:, t].argmax(), t]
            for t in range(pitches.shape[1])
            if pitches[magnitudes[:, t].argmax(), t] > 0
        ]
        mean_pitch = float(np.mean(pitch_values)) if pitch_values else 150.0

        tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo_arr) if np.ndim(tempo_arr) == 0 else float(tempo_arr[0])

        rms = librosa.feature.rms(y=y)[0]
        mean_energy = float(np.mean(rms))

        return {
            "mean_pitch":  round(mean_pitch, 2),
            "tempo":       round(tempo, 2),
            "mean_energy": round(mean_energy, 5),
            "duration":    round(float(len(y) / sr), 2),
        }
    except Exception as e:
        logging.error(f"Prosody extraction error: {e}")
        return defaults


def prosody_to_va(prosody: dict):
    """Convert prosody features to (valence, arousal) tuple."""
    pitch_norm  = np.clip((prosody.get("mean_pitch", 150.0) - 80) / 200, 0, 1)
    energy_norm = np.clip(prosody.get("mean_energy", 0.01) * 100, 0, 1)
    tempo_norm  = np.clip((prosody.get("tempo", 100.0) - 60) / 180, 0, 1)

    valence = round(float(0.4 * pitch_norm + 0.3 * (1 - energy_norm) + 0.3 * tempo_norm), 4)
    arousal = round(float(0.5 * energy_norm + 0.3 * tempo_norm + 0.2 * pitch_norm), 4)
    return valence, arousal

# =============================================================================
# SECTION 8 — EMOTION DETECTION: TEXT
# =============================================================================

_JOY_WORDS = {
    "happy", "happiest", "joy", "joyful", "excited", "amazing", "great",
    "awesome", "fantastic", "excellent", "wonderful", "love", "celebrate",
    "celebrating", "success", "successful", "achieved", "achievement",
    "goal", "won", "win", "best", "good", "pleased", "thrilled",
    "opportunity", "dream", "proud", "pride", "grateful", "thankful",
    "blessed", "fortunate", "lucky", "delighted", "elated", "ecstatic",
    "overjoyed", "cheerful", "optimistic", "hopeful", "energized",
    "inspired", "motivated", "enthusiastic", "passionate", "confident",
    "selected", "promoted", "hired", "offer", "accepted", "reward",
    "accomplished", "milestone", "progress", "growth", "incredible",
    "outstanding", "brilliant", "perfect", "superb", "glorious",
}
_SAD_WORDS = {
    "sad", "cry", "depressed", "upset", "hurt", "lonely", "alone",
    "isolated", "broken", "miss", "pain", "hopeless", "down", "empty",
    "drained", "tired", "burnout",
    "grief", "sorrow", "miserable", "unhappy", "gloomy", "despair",
    "heartbroken", "devastated", "melancholy", "regret", "shame",
    "worthless", "useless", "failure", "defeated", "lost", "stuck",
    "enough", "never", "always", "overwhelmed", "helpless",
}
_ANGER_WORDS = {
    "angry", "furious", "hate", "annoyed", "irritated", "frustrated",
    "rage", "mad", "disgusted", "shouting", "shout",
}
_FEAR_WORDS = {
    "fear", "scared", "afraid", "panic", "worried", "anxious",
    "nervous", "terrified", "unsafe",
}

_PHRASE_BOOSTS = [
    ("emotionally drained",       "sadness",  3.0),
    ("feel alone",                "sadness",  3.0),
    ("completely alone",          "sadness",  3.0),
    ("i feel alone",              "sadness",  3.0),
    ("hurt and disappointed",     "sadness",  5.0),
    ("never seems enough",        "sadness",  5.0),
    ("it never works",            "sadness",  4.0),
    ("no matter how hard",        "sadness",  4.0),
    ("feel like a failure",       "sadness",  5.0),
    ("cannot go on",              "sadness",  5.0),
    ("happiest",                  "joy",      5.0),
    ("dream job",                 "joy",      5.0),
    ("dream opportunity",         "joy",      5.0),
    ("got selected",              "joy",      5.0),
    ("got the offer",             "joy",      5.0),
    ("so happy",                  "joy",      4.0),
    ("really excited",            "joy",      4.0),
    ("feeling great",             "joy",      4.0),
    ("going wonderfully",         "joy",      4.0),
    ("everything is going well",  "joy",      4.0),
    ("frustrated with everyone",  "anger",    5.0),
    ("makes me so angry",         "anger",    5.0),
    ("scared of",                 "fear",     5.0),
    ("terrified of",              "fear",     5.0),
]

_HF_LABEL_MAP = {
    "joy": "joy", "happy": "joy",
    "sadness": "sadness", "sad": "sadness",
    "anger": "anger", "disgust": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "neutral": "neutral",
}


def detect_emotion_text(text: str, text_pipe, return_scores: bool = False):
    """
    Hybrid text emotion detection: rule-based + transformer fusion.
    Returns list of score dicts if return_scores=True,
    else returns (emotion_str, confidence_float).
    """
    _default_neutral = [{"label": "neutral", "score": 1.0}]

    try:
        if not text or not str(text).strip():
            return _default_neutral if return_scores else ("neutral", 1.0)

        text_lower = str(text).lower().strip()

        # --- Rule scores ---
        rule_scores = {"joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "neutral": 0.1}

        for word in text_lower.split():
            word = word.strip(".,!?;:'\"")
            if word in _JOY_WORDS:   rule_scores["joy"]     += 3.0
            if word in _SAD_WORDS:   rule_scores["sadness"] += 3.0
            if word in _ANGER_WORDS: rule_scores["anger"]   += 3.0
            if word in _FEAR_WORDS:  rule_scores["fear"]    += 3.0

        for phrase, emo, boost in _PHRASE_BOOSTS:
            if phrase in text_lower:
                rule_scores[emo] = rule_scores.get(emo, 0.0) + boost

        # --- Transformer scores ---
        model_scores = {"joy": 0.0, "sadness": 0.0, "anger": 0.0,
                        "fear": 0.0, "surprise": 0.0, "neutral": 0.0}

        if text_pipe is not None:
            raw = text_pipe(text)
            if isinstance(raw, list) and raw and isinstance(raw[0], list):
                raw = raw[0]
            for r in raw:
                raw_label = str(r.get("label", "neutral")).lower().strip()
                score = safe_float(r.get("score", 0.0))
                mapped = _HF_LABEL_MAP.get(raw_label, "neutral")
                model_scores[mapped] = model_scores.get(mapped, 0.0) + score

        # --- Hybrid fusion (55% model, 45% rule) ---
        emotions = list(set(list(rule_scores.keys()) + list(model_scores.keys())))
        final_scores = {}
        for emo in emotions:
            m = safe_float(model_scores.get(emo, 0.0))
            r = safe_float(rule_scores.get(emo, 0.0))
            final_scores[emo] = (m * 0.55) + (r * 0.45)

        # Suppress neutral when strong emotion detected
        emotional_strength = max(
            final_scores.get("joy", 0.0),
            final_scores.get("sadness", 0.0),
            final_scores.get("anger", 0.0),
            final_scores.get("fear", 0.0),
        )
        if emotional_strength > 0.35:
            final_scores["neutral"] = final_scores.get("neutral", 0.0) * 0.2

        # Normalize
        total = sum(final_scores.values()) + 1e-9
        scores = [
            {"label": normalize_emotion(k), "score": round(v / total, 4)}
            for k, v in final_scores.items()
        ]
        scores.sort(key=lambda x: x["score"], reverse=True)

        if not scores:
            return _default_neutral if return_scores else ("neutral", 1.0)

        return scores if return_scores else (scores[0]["label"], scores[0]["score"])

    except Exception as e:
        logging.error(f"Text emotion error: {e}")
        return _default_neutral if return_scores else ("neutral", 1.0)

# =============================================================================
# SECTION 9 — EMOTION DETECTION: AUDIO
# =============================================================================

def detect_emotion_audio(wav_path: str, audio_proc, audio_mdl) -> tuple:
    """Returns (emotion_str, confidence_float). Safe — never raises."""
    try:
        if audio_proc is None or audio_mdl is None:
            return "neutral", 0.0

        speech, _ = librosa.load(wav_path, sr=16000)
        if speech is None or len(speech) < 1000:
            return "neutral", 0.0

        inputs = audio_proc(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        device = next(audio_mdl.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = audio_mdl(**inputs).logits

        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())

        raw_label = str(audio_mdl.config.id2label.get(pred_id, "neutral")).lower()
        emotion = normalize_emotion(raw_label)
        confidence = safe_float(probs[pred_id].item())

        del inputs, logits, probs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return emotion, confidence

    except Exception as e:
        logging.error(f"Audio emotion error: {e}")
        gc.collect()
        return "neutral", 0.0

# =============================================================================
# SECTION 10 — MULTIMODAL FUSION MODEL
# =============================================================================

def run_fusion(text_scores, audio_emotion, audio_conf, prosody_features):
    """
    Multimodal fusion logic (text + audio + prosody).
    Returns (final_emotion, fusion_conf, valence, arousal).
    """
    try:
        # ---------------- TEXT EMOTION ----------------
        if text_scores and isinstance(text_scores, list):
            text_best    = max(text_scores, key=lambda x: x["score"])
            text_emotion = normalize_emotion(text_best.get("label", "neutral"))
            text_conf    = safe_float(text_best.get("score", 0.0))
        else:
            text_emotion = "neutral"
            text_conf    = 0.0

        # ---------------- AUDIO ----------------
        audio_emotion = normalize_emotion(audio_emotion)
        audio_conf    = safe_float(audio_conf)

        # ---------------- WEIGHTS ----------------
        prosody_weight = 0.15
        text_weight    = 0.55
        audio_weight   = 0.30

        # Reduce audio weight when signal is weak or neutral
        if audio_conf < 0.2 or audio_emotion == "neutral":
            audio_weight   = 0.10
            text_weight    = 0.75
            prosody_weight = 0.15

        # ---------------- SCORE MAP ----------------
        score_map = {e: 0.0 for e in CANONICAL_EMOTIONS}

        score_map[text_emotion]  += text_conf  * text_weight
        score_map[audio_emotion] += audio_conf * audio_weight

        # Prosody light heuristic
        if prosody_features:
            energy = prosody_features.get("mean_energy", 0.01)
            tempo  = prosody_features.get("tempo", 100.0)
            if energy > 0.02 and tempo > 120:
                score_map["joy"]     += prosody_weight
            elif energy < 0.01:
                score_map["sadness"] += prosody_weight

        # ---------------- FINAL DECISION ----------------
        final_emotion = max(score_map, key=score_map.get)
        fusion_conf   = min(score_map[final_emotion], 1.0)

        # VA from fused emotion defaults
        valence, arousal, _ = get_emotion_va_defaults(final_emotion)

        return final_emotion, fusion_conf, valence, arousal

    except Exception as e:
        logging.error(f"Fusion error: {e}")
        return "neutral", 0.0, 0.5, 0.5


def explain_fusion_decision(text_conf: float, audio_conf: float, audio_emotion: str) -> str:
    """
    Plain-English explanation of what drove the fusion decision.
    Returns a single clean string — rendered once, not duplicated.
    """
    audio_emotion = normalize_emotion(audio_emotion)
    t = safe_float(text_conf)
    a = safe_float(audio_conf)

    if a < 0.20 or audio_emotion == "neutral":
        modality = "Dominated by TEXT modality"
        reason   = (
            "Audio signal is weak or unclear; its contribution to the final "
            "prediction is reduced. Prediction is primarily based on text input."
        )
    elif t >= 0.80 and a < 0.40:
        modality = "Dominated by TEXT modality"
        reason   = (
            f"Text confidence is very high ({t:.0%}) while audio confidence is "
            f"low ({a:.0%}). The text signal drives the final decision."
        )
    elif a >= 0.50 and t < 0.60:
        modality = "Dominated by AUDIO modality"
        reason   = (
            f"Audio confidence ({a:.0%}) is notably high relative to text "
            f"confidence ({t:.0%}). Voice signals carry significant weight here."
        )
    else:
        modality = "Balanced TEXT + AUDIO contribution"
        reason   = (
            f"Both text ({t:.0%}) and audio ({a:.0%}) signals contribute "
            "meaningfully to the fusion decision."
        )

    return f"**Fusion Decision:** {modality}. {reason}"

# =============================================================================
# SECTION 11 — LIME EXPLAINABILITY
# =============================================================================

def get_lime_explanation(text: str, text_pipe, lime_explainer) -> str:
    """
    LIME-based text explainability.
    FIX: narrative no longer hardcodes 'positive emotional signals' for all labels;
    it now uses emotion-appropriate language.
    """
    try:
        if lime_explainer is None or text_pipe is None:
            return "LIME unavailable (explainer or model not loaded)"
        if not text or not str(text).strip():
            return "No text provided for explanation"

        def predict_proba(texts: list) -> np.ndarray:
            results = []
            for t in texts:
                scores_list = detect_emotion_text(t, text_pipe, return_scores=True)
                score_map = {
                    normalize_emotion(item["label"]): safe_float(item["score"])
                    for item in scores_list
                }
                row = np.array(
                    [score_map.get(emo, 0.0) for emo in CANONICAL_EMOTIONS],
                    dtype=np.float32,
                )
                total = row.sum()
                if total > 0:
                    row = row / total
                results.append(row)
            return np.array(results, dtype=np.float32)

        explanation = lime_explainer.explain_instance(
            str(text).strip(),
            predict_proba,
            num_features=6,
            num_samples=100,
            top_labels=1,
        )

        top_label_idx  = explanation.top_labels[0]
        top_label_name = CANONICAL_EMOTIONS[top_label_idx].upper()
        word_weights   = explanation.as_list(label=top_label_idx)

        if not word_weights:
            return f"LIME: No influential words found for label '{top_label_name}'"

        positive_pairs = sorted([(w, v) for w, v in word_weights if v > 0],
                                key=lambda x: x[1], reverse=True)
        negative_pairs = sorted([(w, v) for w, v in word_weights if v <= 0],
                                key=lambda x: x[1])

        lines = [f"── LIME Explanation  ·  Predicted Label: {top_label_name} ──\n"]

        if positive_pairs:
            lines.append("▲ Words Supporting This Prediction:")
            for word, weight in positive_pairs:
                bar = "█" * min(int(abs(weight) * 40), 12)
                lines.append(f"   {word:<15}  {weight:+.4f}  {bar}")
        else:
            lines.append("▲ No strong positive contributors found.")

        lines.append("")

        if negative_pairs:
            lines.append("▼ Words Reducing Confidence:")
            for word, weight in negative_pairs:
                bar = "░" * min(int(abs(weight) * 40), 12)
                lines.append(f"   {word:<15}  {weight:+.4f}  {bar}")
        else:
            lines.append("▼ No suppressing words detected.")

        lines.append("")

        # --- FIX: emotion-aware narrative (no longer always says "positive emotional signals") ---
        pos_words  = [w for w, _ in positive_pairs]
        neg_words  = [w for w, _ in negative_pairs]
        total_pos  = sum(v for _, v in positive_pairs)
        total_neg  = abs(sum(v for _, v in negative_pairs))
        ratio      = total_pos / (total_pos + total_neg + 1e-9)

        # Emotion-specific signal descriptors
        _signal_desc = {
            "JOY":      "positive emotional signals",
            "SADNESS":  "negative/low-valence emotional signals",
            "ANGER":    "high-arousal negative emotional signals",
            "FEAR":     "anxiety or threat-related signals",
            "SURPRISE": "unexpected or novel-event signals",
            "RELAXED":  "calm and low-arousal signals",
            "NEUTRAL":  "ambiguous or low-intensity signals",
        }
        signal_desc = _signal_desc.get(top_label_name, "emotional signals")

        if pos_words:
            top_pos   = ", ".join(f'"{w}"' for w in pos_words[:3])
            narrative = (
                f"📌 The model predicts {top_label_name} due to strong "
                f"{signal_desc} from words such as {top_pos}."
            )
        else:
            narrative = f"📌 The model predicts {top_label_name} without strong lexical anchors."

        if ratio >= 0.80:
            strength = "⚡ Signal Strength: STRONG — highly aligned contributions."
        elif ratio >= 0.55:
            strength = "⚠️  Signal Strength: MODERATE — mixed signals present."
        else:
            strength = "🔻 Signal Strength: WEAK — opposing signals reduce certainty."

        if neg_words:
            top_neg  = ", ".join(f'"{w}"' for w in neg_words[:2])
            counter  = f"Counter-signal words ({top_neg}) slightly reduce confidence."
        else:
            counter  = "No significant counter-signal words detected."

        lines.append(narrative)
        lines.append(strength)
        lines.append(f"   {counter}")

        return "\n".join(lines)

    except Exception as e:
        logging.error(f"LIME explanation error: {e}")
        return f"LIME error: {e}"

# =============================================================================
# SECTION 12 — AUXILIARY EMOTION INTELLIGENCE
# =============================================================================

def simulate_fatigue(text: str, prosody: dict = None) -> str:
    fatigue_lexicon = {"exhausted", "tired", "burned out", "burnt out", "drained", "sleepy"}
    text_lower = (text or "").lower()
    if any(w in text_lower for w in fatigue_lexicon):
        return "Fatigue (Lexical)"
    if prosody:
        if prosody.get("mean_energy", 1.0) < 0.01 and prosody.get("tempo", 100.0) < 90:
            return "Fatigue (Voice)"
    return "Alert"


def estimate_empathy_level(text: str, memory) -> str:
    emotional_keywords = [
        "hurt", "alone", "sad", "cry", "depressed", "anxious", "stress",
        "fear", "scared", "panic", "frustrated", "angry", "upset", "tired",
        "exhausted", "broken", "hopeless", "nobody", "pain", "lost",
    ]
    text_lower = (text or "").lower()
    score = sum(text_lower.count(w) for w in emotional_keywords)
    for msg in list(memory)[-5:]:
        if msg:
            m = str(msg).lower()
            score += sum(m.count(w) for w in emotional_keywords)
    if score >= 5:
        return "High"
    elif score >= 2:
        return "Medium"
    return "Low"


def get_cbt_prompt(emotion: str, intensity: float, valence: float, empathy: str) -> str:
    emotion = normalize_emotion(emotion or "neutral")
    empathy = (empathy or "Low").lower().strip()

    templates = {
        
    "sadness": {
        "low": (
            "I’m here with you. Difficult moments happen, but they do not define your entire day. "
            "What is one small thing that brought even a little comfort or peace today?"
        ),
        "medium": (
            "I understand this feels emotionally heavy right now. Remember, feelings can change with time and support. "
            "What is one gentle step you can take today to care for yourself?"
        ),
        "high": (
            "This seems deeply overwhelming, and your feelings deserve kindness, not judgment. "
            "Pause for a moment and focus on your breathing. Notice 3 safe or comforting things around you right now. "
            "You do not have to solve everything at once."
        ),
    },

    "anger": {
        "low": (
            "It sounds like something disturbed your peace a little. "
            "What part of the situation can you respond to calmly and confidently?"
        ),
        "medium": (
            "Your frustration is understandable, but your emotions do not control your actions. "
            "What would help you feel more balanced and heard in this moment?"
        ),
        "high": (
            "This emotion feels very intense right now. Before reacting, give yourself permission to pause. "
            "Take one slow breath and focus only on what is within your control today. "
            "Protecting your peace is important."
        ),
    },

    "fear": {
        "low": (
            "That concern sounds important, and it’s okay to feel uncertain sometimes. "
            "What is one reassuring thought or fact that can help you feel safer right now?"
        ),
        "medium": (
            "Fear can make situations feel larger than they are. "
            "Try focusing on one small step forward instead of the whole problem at once."
        ),
        "high": (
            "This feels emotionally overwhelming, but you are not alone in this moment. "
            "Ground yourself gently — focus on your breathing and remind yourself: "
            "you only need to handle the next small step, not everything at once."
        ),
    },

    "surprise": {
        "low": (
            "Unexpected moments can bring new perspectives. "
            "What positive possibility could come from this situation?"
        ),
        "medium": (
            "This situation shifted your expectations, and that’s okay. "
            "Sometimes change opens paths we had not considered before."
        ),
        "high": (
            "This feels like a major emotional shift. "
            "Give yourself time to process it slowly and calmly. "
            "Even sudden changes can lead to growth, clarity, or new opportunities."
        ),
    },

    "joy": {
        "low": (
            "It’s wonderful to notice positive moments, even small ones. "
            "What helped create this feeling today?"
        ),
        "medium": (
            "You seem to be in a healthy emotional space right now. "
            "Recognizing what supports your happiness can help you build more of these moments."
        ),
        "high": (
            "This is a beautiful positive emotional state. "
            "Take a moment to appreciate it fully and think about how you can nurture this feeling again in the future."
        ),
    },

    "relaxed": {
        "low": (
            "You seem calm and steady right now. "
            "What is helping you maintain this sense of balance?"
        ),
        "medium": (
            "This emotional stability is valuable. "
            "Healthy routines and mindful habits often create moments like this."
        ),
        "high": (
            "You’re in a very peaceful state emotionally. "
            "This is a great moment for gratitude, reflection, or setting positive intentions for yourself."
        ),
    },

    "neutral": {
        "low": (
            "You seem emotionally balanced at the moment. "
            "Is there anything meaningful or important you’ve been thinking about recently?"
        ),
        "medium": (
            "Neutral moments can sometimes be opportunities for self-reflection. "
            "What is one thing you would like to improve or focus on moving forward?"
        ),
        "high": (
            "A calm emotional state can be very powerful. "
            "This may be a good time to reflect on your goals, progress, and the positive habits supporting you."
        ),
    }
    }

    if emotion not in templates:
        emotion = "neutral"

    level    = "high" if intensity >= 0.7 else ("low" if intensity < 0.4 else "medium")
    response = templates[emotion][level]

    empathy_add = {
        "high":   " I want you to know that your feelings are completely valid.",
        "medium": " I'm here with you through this.",
        "low":    "",
    }
    response += empathy_add.get(empathy, "")

    if valence < 0.4 and emotion in {"sadness", "fear", "anger"}:
        response += " Let's take this step by step."
    if valence > 0.6 and emotion == "joy":
        response += " Keep building on this positive momentum."

    return response


def generate_empathetic_response(emotion: str) -> str:
    emotion = normalize_emotion(emotion)
    responses = {
        "joy":      "😊 That's really nice! Sounds like something good happened. Want to share more?",
        "sadness":  "💙 That sounds difficult. I'm here with you. What's been bothering you the most?",
        "anger":    "😠 I can feel the frustration. What exactly triggered this feeling?",
        "fear":     "😟 That seems stressful. Let's break it down — what part worries you the most?",
        "surprise": "😲 That's unexpected! Was it a good surprise or something confusing?",
        "relaxed":  "😌 You seem calm. That's a great state. What's helping you feel this way?",
        "neutral":  "🙂 I understand. Tell me a bit more so I can better understand you.",
    }
    return responses.get(emotion, "🙂 I understand. Tell me a bit more.")

# =============================================================================
# SECTION 13 — TEMPORAL DRIFT ANALYSIS
# =============================================================================

_TRANSITION_MAP = {
    ("sadness", "anger"):   {"label": "Emotional Escalation",          "description": "Sadness transformed into frustration/anger.",      "risk": "Medium"},
    ("fear",    "anger"):   {"label": "Defensive Escalation",          "description": "Fear response escalated into anger.",               "risk": "High"},
    ("neutral", "anger"):   {"label": "Sudden Emotional Activation",   "description": "Rapid activation of negative emotional state.",    "risk": "Medium"},
    ("fear",    "neutral"): {"label": "Recovery",                       "description": "Fear reduced toward emotional stability.",          "risk": "Low"},
    ("sadness", "neutral"): {"label": "Emotional Stabilization",        "description": "Negative emotional intensity reduced.",            "risk": "Low"},
    ("anger",   "relaxed"): {"label": "Successful Emotional Regulation","description": "Anger successfully reduced into calmness.",        "risk": "Low"},
    ("neutral", "joy"):     {"label": "Positive Emotional Shift",       "description": "Movement toward positive emotional state.",        "risk": "Low"},
    ("sadness", "joy"):     {"label": "Strong Positive Recovery",       "description": "Significant emotional improvement detected.",      "risk": "Low"},
    ("sadness", "sadness"): {"label": "Persistent Sadness",             "description": "Negative emotional state remains stable.",         "risk": "Medium"},
    ("fear",    "fear"):    {"label": "Persistent Anxiety",             "description": "Fear/anxiety sustained over time.",                "risk": "High"},
    ("anger",   "anger"):   {"label": "Persistent Anger",               "description": "Anger maintained across interactions.",            "risk": "High"},
}

_DEFAULT_TRANSITION = {
    "label":       "Normal Emotional Variation",
    "description": "Natural emotional fluctuation detected.",
    "risk":        "Low",
}


def analyze_emotion_drift(prev: str, curr: str) -> dict:
    return _TRANSITION_MAP.get(
        (normalize_emotion(prev), normalize_emotion(curr)),
        _DEFAULT_TRANSITION,
    )


def compute_transition_probability(history: list, prev: str, curr: str) -> float:
    if len(history) < 2:
        return 0.0
    transitions  = [(history[i], history[i + 1]) for i in range(len(history) - 1)]
    total_prev   = sum(1 for t in transitions if t[0] == prev)
    if total_prev == 0:
        return 0.0
    matching = sum(1 for t in transitions if t == (prev, curr))
    return round(matching / total_prev, 3)


def compute_emotion_stability(history: list) -> float:
    if len(history) < 2:
        return 1.0
    changes = sum(1 for i in range(len(history) - 1) if history[i] != history[i + 1])
    return round(1 - changes / (len(history) - 1), 2)

# =============================================================================
# SECTION 14 — STREAMLIT PAGE CONFIG & STYLING
# =============================================================================

st.set_page_config(page_title="🎙️ Emotion AI Assistant", layout="wide")

st.markdown("""
<style>
.main { background-color: #f0f4f8; }
.stButton > button {
    background-color: #4CAF50; color: white;
    font-weight: bold; border-radius: 10px;
}
.stTextInput > div > input { border-radius: 10px; }
.stTextArea textarea { border-radius: 10px; }
.block-container { padding: 2rem; }
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem; border-radius: 10px; color: white;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SECTION 15 — INITIALIZE DB & MODELS
# =============================================================================

init_db()

whisper_model                 = load_whisper_model()
text_pipeline                 = load_text_model()
audio_processor, audio_model  = load_audio_model()
lime_explainer                = load_lime_explainer()

# =============================================================================
# SECTION 16 — SESSION STATE INITIALIZATION
# =============================================================================

def _init_session():
    defaults = {
        "messages":            deque(maxlen=20),
        "wav_path":            None,
        "transcribed_text":    "",
        "text_input_field":    "",
        "audio_features":      None,
        "emotion_history":     deque(maxlen=30),
        "emotion_transitions": [],
        "results":             None,
        "warmup_done":         False,
        "last_emotion":        "neutral",
        "last_confidence":     0.0,
        "app_errors":          [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset_session():
    """Hard-reset all user-facing state keys in-place. warmup_done is preserved."""
    st.session_state.messages             = deque(maxlen=20)
    st.session_state.wav_path             = None
    st.session_state.transcribed_text     = ""
    st.session_state["text_input_field"]  = ""
    st.session_state.audio_features       = None
    st.session_state.emotion_history      = deque(maxlen=30)
    st.session_state.emotion_transitions  = []
    st.session_state.results              = None
    st.session_state.last_emotion         = "neutral"
    st.session_state.last_confidence      = 0.0
    st.session_state.app_errors           = []


_init_session()

if not st.session_state.warmup_done:
    warmup_models(text_pipeline, audio_processor, audio_model)
    st.session_state.warmup_done = True

if st.session_state.get("reset_flag"):
    st.session_state.reset_flag = False
    _reset_session()
    st.rerun()

# =============================================================================
# SECTION 17 — MAIN UI HEADER
# =============================================================================

st.title("🧠 Emotion AI Assistant")
st.divider()

# =============================================================================
# SECTION 18 — INPUT PANEL (Record + Upload + Text)
# =============================================================================

col_rec, col_upload = st.columns([2, 1])

with col_rec:
    with st.expander("🎤 Record Audio (5 seconds)"):
        if st.button("🎙️ Record Now", key="record_btn"):
            samplerate = 16000
            duration   = 5

            try:
                sd.check_input_settings(samplerate=samplerate, channels=1)
            except Exception as e:
                st.warning(f"Audio device warning: {e}")

            recording = None
            wav_path  = None

            try:
                with st.spinner("🔴 Recording — please speak now…"):
                    recording = sd.rec(
                        int(samplerate * duration),
                        samplerate=samplerate,
                        channels=1,
                        dtype="float32",
                    )
                    sd.wait()
            except Exception as e:
                st.error(f"Microphone error: {e}")
                recording = np.zeros((samplerate * duration, 1), dtype=np.float32)

            try:
                recording = np.squeeze(recording)
                recording = np.nan_to_num(np.clip(recording, -1.0, 1.0))
                ts = datetime.datetime.now().timestamp()
                wav_path = os.path.join(tempfile.gettempdir(), f"rec_{ts}.wav")
                wavio.write(wav_path, recording, samplerate, sampwidth=2)
                st.session_state.wav_path = wav_path
            except Exception as e:
                st.error(f"Audio save error: {e}")
                st.session_state.wav_path = None

            try:
                if wav_path and whisper_model:
                    result = whisper_model.transcribe(wav_path)
                    text = result.get("text", "").strip() if isinstance(result, dict) else ""
                    st.session_state.transcribed_text    = text
                    st.session_state["text_input_field"] = text
                    if text:
                        st.success(f"✅ Transcribed: {text}")
                    else:
                        st.warning("⚠️ No speech detected clearly.")
                    del result
                    gc.collect()
                else:
                    st.session_state.transcribed_text    = ""
                    st.session_state["text_input_field"] = ""
                    st.warning("⚠️ Whisper not loaded — please type manually.")
            except Exception as e:
                st.session_state.transcribed_text    = ""
                st.session_state["text_input_field"] = ""
                st.error(f"Transcription error: {e}")
                gc.collect()

with col_upload:
    with st.expander("📁 Upload WAV File"):
        audio_file = st.file_uploader(
            "Choose a WAV file", type=["wav"],
            help="200MB per file • WAV format only"
        )
        if audio_file is not None:
            ts = datetime.datetime.now().timestamp()
            wav_path = os.path.join(tempfile.gettempdir(), f"upload_{ts}.wav")
            with open(wav_path, "wb") as f:
                f.write(audio_file.read())
            st.session_state.wav_path = wav_path
            try:
                if whisper_model:
                    result      = whisper_model.transcribe(wav_path)
                    transcribed = result.get("text", "").strip() if isinstance(result, dict) else ""
                    st.session_state.transcribed_text    = transcribed
                    st.session_state["text_input_field"] = transcribed
                    if transcribed:
                        st.success(f"✅ Transcribed: {transcribed}")
                    else:
                        st.warning("⚠️ No speech detected. You may type manually.")
                else:
                    st.session_state.transcribed_text    = ""
                    st.session_state["text_input_field"] = ""
                    st.success("✅ Audio uploaded (Whisper unavailable — type manually)")
            except Exception as e:
                st.session_state.transcribed_text    = ""
                st.session_state["text_input_field"] = ""
                st.error(f"Transcription error: {e}")

st.subheader("✍️ Text Input")
final_text = st.text_input(
    "Enter your message (or use recorded text above)",
    key="text_input_field",
)

if st.session_state.wav_path and os.path.exists(str(st.session_state.wav_path)):
    st.audio(st.session_state.wav_path)

# =============================================================================
# SECTION 19 — ANALYSIS TRIGGER
# =============================================================================

if st.button("🔍 Analyze Emotion", type="primary", key="analyze_btn"):

    if not str(final_text).strip():
        st.error("⚠️ Please enter some text or record audio first.")
    else:
        with st.spinner("🔬 Running multimodal emotion analysis…"):

            # --- TEXT ---
            text_scores = detect_emotion_text(final_text, text_pipeline, return_scores=True)
            if not text_scores:
                text_scores = [{"label": "neutral", "score": 1.0}]

            text_best    = max(text_scores, key=lambda x: x["score"])
            text_emotion = normalize_emotion(text_best.get("label", "neutral"))
            text_conf    = safe_float(text_best.get("score", 0.0))

            # --- AUDIO + PROSODY ---
            audio_emotion    = "neutral"
            audio_conf       = 0.0
            prosody_features = {}

            wav_path = st.session_state.get("wav_path")
            if wav_path and os.path.exists(str(wav_path)):
                prosody_features = extract_prosody_features(wav_path)
                st.session_state.audio_features = prosody_features
                audio_emotion, audio_conf = detect_emotion_audio(
                    wav_path, audio_processor, audio_model
                )

            # --- FUSION ---
            fusion_emotion, fusion_conf, fused_valence, fused_arousal = run_fusion(
                text_scores, audio_emotion, audio_conf, prosody_features
            )
            final_emotion = normalize_emotion(fusion_emotion)

            # VA: use prosody-derived VA when audio exists; else use emotion defaults
            if wav_path and os.path.exists(str(wav_path)) and prosody_features:
                valence, arousal = prosody_to_va(prosody_features)
            else:
                valence, arousal, _ = get_emotion_va_defaults(final_emotion)

            # --- INTENSITY ---
            intensity = calculate_intensity(valence, arousal)

            # --- TEMPORAL DRIFT ---
            prev_emotion      = None
            transition_result = None
            transition_prob   = 0.0

            history = list(st.session_state.emotion_history)
            if history:
                prev_emotion      = history[-1]
                transition_result = analyze_emotion_drift(prev_emotion, final_emotion)
                transition_prob   = compute_transition_probability(history, prev_emotion, final_emotion)

            st.session_state.emotion_history.append(final_emotion)
            stability_score = compute_emotion_stability(list(st.session_state.emotion_history))

            # --- AUXILIARY ---
            fatigue            = simulate_fatigue(final_text, prosody_features)
            empathy            = estimate_empathy_level(final_text, st.session_state.messages)
            cbt                = get_cbt_prompt(final_emotion, intensity, valence, empathy)
            shap_exp           = get_lime_explanation(final_text, text_pipeline, lime_explainer)
            response           = generate_empathetic_response(final_emotion)
            fusion_explanation = explain_fusion_decision(text_conf, audio_conf, audio_emotion)

            # --- MEMORY ---
            if final_text.strip():
                st.session_state.messages.append(final_text.strip())

            # --- FUSION CONFIDENCE (safe) ---
            fusion_conf_db = min(
                max((text_conf * 0.60) + (audio_conf * 0.25) + 0.15, 0.0), 1.0
            )

            # --- DB INSERT ---
            try:
                insert_emotion_record((
                    datetime.datetime.now().isoformat(),
                    final_emotion,
                    fatigue,
                    empathy,
                    intensity,
                    final_text,
                    cbt,
                    valence,
                    arousal,
                    text_conf,
                    audio_conf,
                    fusion_conf_db,
                    shap_exp,
                ))
            except Exception as e:
                st.error(f"Database insert error: {e}")

            logging.info(
                f"emotion={final_emotion} text_conf={text_conf:.3f} "
                f"audio_conf={audio_conf:.3f} valence={valence:.3f} "
                f"arousal={arousal:.3f} intensity={intensity:.3f} empathy={empathy}"
            )

            # --- STORE RESULTS ---
            st.session_state.results = {
                "text_emotion":       text_emotion,
                "text_conf":          text_conf,
                "audio_emotion":      audio_emotion,
                "audio_conf":         audio_conf,
                "fusion_emotion":     fusion_emotion,
                "fusion_conf":        fusion_conf,
                "final_emotion":      final_emotion,
                "decision_basis":     "Multimodal Fusion",
                "valence":            valence,
                "arousal":            arousal,
                "intensity":          intensity,
                "fatigue":            fatigue,
                "empathy":            empathy,
                "cbt":                cbt,
                "shap":               shap_exp,
                "fusion_explanation": fusion_explanation,
                "response":           response,
                "prosody":            prosody_features,
                "prev_emotion":       prev_emotion,
                "transition_result":  transition_result,
                "transition_prob":    transition_prob,
                "stability_score":    stability_score,
                # FIX: flag so UI knows whether audio was present (avoid duplicate caption)
                "audio_present":      bool(wav_path and os.path.exists(str(wav_path))),
            }

# =============================================================================
# SECTION 20 — RESULTS DISPLAY
# =============================================================================

results = st.session_state.get("results")

if results and isinstance(results, dict) and results.get("final_emotion"):

    final_emotion   = results["final_emotion"]
    valence         = safe_float(results.get("valence", 0.5))
    arousal         = safe_float(results.get("arousal", 0.5))
    intensity       = safe_float(results.get("intensity", 0.0))
    stability_score = safe_float(results.get("stability_score", 1.0))

    text_emotion    = results.get("text_emotion", "neutral")
    text_conf       = safe_float(results.get("text_conf", 0.0))
    audio_emotion   = results.get("audio_emotion", "neutral")
    audio_conf      = safe_float(results.get("audio_conf", 0.0))
    fusion_emotion  = results.get("fusion_emotion", "neutral")
    fusion_conf     = safe_float(results.get("fusion_conf", 0.0))

    fatigue             = results.get("fatigue", "Alert")
    empathy             = results.get("empathy", "Low")
    cbt                 = results.get("cbt", "")
    shap_exp            = results.get("shap", "")
    response            = results.get("response", "")
    prosody             = results.get("prosody", {})
    audio_present       = results.get("audio_present", False)
    fusion_explanation  = results.get("fusion_explanation", "")
    decision_basis      = results.get("decision_basis", "Fusion Model")

    prev_emotion       = results.get("prev_emotion")
    transition_result  = results.get("transition_result")
    transition_prob    = safe_float(results.get("transition_prob", 0.0))

    # --- TOP METRICS ---
    st.subheader("📊 Analysis Results")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🎭 Emotion",   final_emotion.title())
    m2.metric("📊 Valence",   f"{valence:.3f}")
    m3.metric("🔥 Arousal",   f"{arousal:.3f}")
    m4.metric("⚡ Intensity", f"{intensity:.3f}")
    m5.metric("🧠 Stability", f"{stability_score:.2f}")
    st.caption(f"Method: {decision_basis}")

    st.divider()

    # --- MODEL BREAKDOWN ---
    c1, c2, c3 = st.columns(3)
    c1.info(f"📝 **Text:** {text_emotion} ({text_conf:.1%})")
    c2.info(f"🎵 **Audio:** {audio_emotion} ({audio_conf:.1%})")
    c3.warning(f"🔗 **Fusion:** {fusion_emotion} ({fusion_conf:.1%})")
    st.progress(min(max(fusion_conf, 0.0), 1.0))

    # FIX: single unified fusion note — rendered exactly once, never duplicated.
    # Audio-weak note is embedded inside fusion_explanation when appropriate,
    # so we do NOT render a separate st.caption for the audio signal.
    # FIX: show fusion explanation but REMOVE repetitive "Dominated by TEXT modality" line
    if fusion_explanation:
        cleaned_explanation = fusion_explanation.replace(
            "Dominated by TEXT modality.",
            ""
    ).strip()

    st.markdown(cleaned_explanation)
    st.divider()

    # --- CBT + AUXILIARY ---
    st.subheader("💡 Insight & Support")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"**Fatigue State:** `{fatigue}`")
        st.markdown(f"**Empathy Level:** `{empathy}`")
        st.markdown(f"**💭 CBT Prompt:**\n\n_{cbt}_")

    with col_b:
        st.markdown("**🔍 LIME Explainability**")
        if shap_exp:
            st.code(shap_exp, language=None)
        # Prosody shown only when audio was actually analyzed
        if audio_present and isinstance(prosody, dict) and prosody:
            st.markdown("**🎵 Prosody Features:**")
            st.json(prosody)

    # --- TRANSITION ANALYSIS (only when previous emotion exists) ---
    if prev_emotion and transition_result:
        st.divider()
        risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(
            transition_result.get("risk", "Low"), "🟡"
        )
        st.subheader("📉 Emotional Transition Analysis")
        tr1, tr2, tr3 = st.columns(3)
        tr1.metric("Previous Emotion", prev_emotion.title())
        tr2.metric("Current Emotion",  final_emotion.title())
        tr3.metric("Transition Probability", f"{transition_prob:.1%}")
        st.markdown(
            f"{risk_color} **{transition_result.get('label', '')}** — "
            f"{transition_result.get('description', '')} "
            f"*(Risk: {transition_result.get('risk', 'Low')})*"
        )

    st.divider()

    # --- ASSISTANT RESPONSE ---
    st.success(f"🤖 Assistant: {response}")

else:
    st.info("🔍 Run analysis above to view results.")

# =============================================================================
# SECTION 21 — DASHBOARD (single source of truth: DB)
# =============================================================================

st.divider()
st.header("📈 Session Dashboard")

db_data = load_db_records(200)

if db_data.empty:
    st.info("📭 No analysis records yet. Run an analysis to populate the dashboard.")
else:
    db_data["timestamp"] = pd.to_datetime(db_data["timestamp"], errors="coerce")
    db_data = db_data.dropna(subset=["timestamp"])
    db_data["emotion"] = db_data["emotion"].apply(normalize_emotion)

    dash_col1, dash_col2 = st.columns(2)

    with dash_col1:
        st.subheader("📊 Emotion Frequency (Absolute Count)")
        emotion_counts = db_data["emotion"].value_counts()
        fig_bar = px.bar(
            x=emotion_counts.index,
            y=emotion_counts.values,
            color=emotion_counts.index,
            color_discrete_map=EMOTION_COLORS,
            labels={"x": "Emotion", "y": "Count"},
            title=f"Total Occurrences per Emotion (n={len(db_data)})",
            text_auto=True,
        )
        fig_bar.update_layout(template="plotly_white", height=380, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True, key="dash_bar_chart")
        st.metric("Most Common Emotion",
                  f"{emotion_counts.idxmax().title()} ({emotion_counts.max()})")

    with dash_col2:
        st.subheader("🥧 Emotion Distribution (Percentage Share)")
        fig_pie = px.pie(
            values=emotion_counts.values,
            names=emotion_counts.index,
            color=emotion_counts.index,
            color_discrete_map=EMOTION_COLORS,
            title="Proportion of Each Emotion",
        )
        fig_pie.update_layout(height=380)
        st.plotly_chart(fig_pie, use_container_width=True, key="dash_pie_chart")

    # --- EMOTION TREND OVER TIME ---
    st.subheader("📉 Emotion Trend Over Time")
    if "timestamp" in db_data.columns and len(db_data) >= 2:
        trend_df = db_data[["timestamp", "emotion"]].dropna().sort_values("timestamp")
        emotion_order = CANONICAL_EMOTIONS
        emotion_rank  = {e: i for i, e in enumerate(emotion_order)}
        trend_df = trend_df.copy()
        trend_df["emotion_rank"] = trend_df["emotion"].map(emotion_rank).fillna(0)
        trend_df["label"]        = trend_df["emotion"].str.title()

        fig_trend = px.line(
            trend_df,
            x="timestamp",
            y="emotion_rank",
            markers=True,
            hover_data={"label": True, "emotion_rank": False, "timestamp": True},
            title="Emotion Progression Over Session",
            labels={"timestamp": "Time", "emotion_rank": "Emotion"},
            color_discrete_sequence=["#667eea"],
        )
        fig_trend.update_yaxes(
            tickvals=list(emotion_rank.values()),
            ticktext=[e.title() for e in emotion_order],
        )
        fig_trend.update_traces(
            mode="lines+markers+text",
            text=trend_df["label"],
            textposition="top center",
            marker=dict(size=10),
            line=dict(width=2.5),
        )
        fig_trend.update_layout(
            template="plotly_white",
            height=380,
            hovermode="x unified",
            showlegend=False,
        )
        st.plotly_chart(fig_trend, use_container_width=True, key="dash_trend_chart")
        st.caption(
            "Each point is one analysis event. "
            "The vertical axis shows emotion category; "
            "the horizontal axis shows time of analysis."
        )
    else:
        st.info("📭 At least 2 records required to plot emotion trend over time.")

    # --- RECENT RECORDS ---
    st.subheader("📝 Recent Analyses")
    preview_cols = [
        "timestamp", "emotion", "valence", "arousal", "intensity",
        "fatigue", "empathy", "text_confidence", "audio_confidence", "fusion_confidence",
    ]
    available = [c for c in preview_cols if c in db_data.columns]
    st.dataframe(
        db_data[available].head(15).reset_index(drop=True),
        use_container_width=True,
        height=380,
    )

# =============================================================================
# SECTION 22 — RESET & DEBUG PANEL
# =============================================================================

st.divider()

with st.expander("🔄 Reset & Debug", expanded=False):

    st.subheader("🧠 Session Memory (last 10)")
    messages = list(st.session_state.get("messages", []))

    if messages:
        for i, msg in enumerate(reversed(messages[-10:]), start=1):
            st.write(f"{i}. {msg}")
    else:
        st.info("No session memory available.")

    st.divider()

    st.subheader("🛠️ System Status Overview")
    st.caption(
        "This section represents system runtime status "
        "and does not include user data."
    )

    gpu_ok = bool(torch.cuda.is_available())

    status_rows = [
        ("GPU Status",           "✅ Available"  if gpu_ok else "❌ Not Available"),
        ("Execution Device",     f"🖥️ {DEVICE.upper()}"),
        ("Whisper (ASR)",        "✅ Loaded"  if whisper_model  is not None else "❌ Not Loaded"),
        ("Text Emotion Model",   "✅ Loaded"  if text_pipeline  is not None else "❌ Not Loaded"),
        ("Audio Emotion Model",  "✅ Loaded"  if audio_model    is not None else "❌ Not Loaded"),
        ("LIME Explainability",  "✅ Active"  if lime_explainer is not None else "❌ Inactive"),
        ("Database",             "✅ Present" if os.path.exists(DB_PATH) else "❌ Missing"),
        (
            "Audio File (session)",
            "✅ Present"
            if (
                st.session_state.get("wav_path")
                and os.path.exists(str(st.session_state.get("wav_path", "")))
            )
            else "— None",
        ),
        ("Session Messages",     str(len(st.session_state.get("messages",        [])))),
        ("Emotion History Len",  str(len(st.session_state.get("emotion_history", [])))),
    ]

    status_df = pd.DataFrame(status_rows, columns=["Component", "Status"])
    st.dataframe(status_df, use_container_width=True, hide_index=True, height=390)

    st.divider()

    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("🧠 Clear Session", key="btn_clear_session"):
            try:
                wav = st.session_state.get("wav_path")
                if wav and os.path.exists(str(wav)):
                    try:
                        os.remove(wav)
                    except Exception:
                        pass

                st.session_state.reset_flag = True
                st.toast("Session clearing... 🔄")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Clear session error: {e}")

    with btn_col2:
        if st.button("🗑️ Clear Database", key="btn_clear_db"):
            try:
                conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
                conn.execute("DELETE FROM emotions")
                conn.commit()
                conn.close()
                st.success("✅ Database cleared.")
                st.rerun()
            except Exception as e:
                st.error(f"Database clear error: {e}")

    st.divider()

    # FIX: DB preview is entirely inside the expander — no bleed into main body
    st.subheader("🗃️ Database Preview (latest 5)")
    preview = load_db_records(5)

    if preview.empty:
        st.info("Database is empty.")
    else:
        st.dataframe(preview, use_container_width=True, height=220)