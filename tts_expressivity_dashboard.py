import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile, os, io
from dotenv import load_dotenv
import time


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TTS Expressivity Lab",
    page_icon="🎙️",
    layout="wide",
)


from hume import HumeClient
load_dotenv()

def get_secret(key):
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        pass
    return os.getenv(key)

from hume.expression_measurement.batch.types import Models, Prosody, InferenceBaseRequest
HUME_API_KEY = get_secret('HUME_API_KEY')

if not HUME_API_KEY:
    st.error("Missing HUME_API_KEY in .env file!")
    st.stop()

client = HumeClient(api_key=HUME_API_KEY)


# ══════════════════════════════════════════════════════════════════════════════
# EMOTION CONFIGURATION
# Maps emotion label → scoring parameters for valence and Hume.
# valence_direction: "negative" → lower valence scores better
#                   "positive" → higher valence scores better
# hume_emotion: exact emotion label string as returned by Hume API
# ══════════════════════════════════════════════════════════════════════════════
EMOTION_CONFIG = {
    #I want description to not do anything but I'm not deleting it until I know if it is used anywhere
    "Anger": {
        "valence_direction": "negative",
        "hume_emotion": "Anger",
        "description": "High arousal, negative valence. Anger should sound forceful and tense.",
    },
    "Excitement": {
        "valence_direction": "positive",
        "hume_emotion": "Excitement",
        "description": "High arousal, positive valence.",
    },
    "Interest": {
        "valence_direction": "positive",
        "hume_emotion": "Interest",
        "description": "Low arousal, positive valence.",
    },
    "Sympathy": {
        "valence_direction": "positive", #I don't know
        "hume_emotion": "Sympathy",
        "description": "medium?? arousal, medium?? valence.",
    },
    "Surprise": {
        "valence_direction": "positive",
        "hume_emotion": "Surprise (positive)",
        "description": "High arousal, positive valence.",
    },
    "Contempt": {
        "valence_direction": "negative",
        "hume_emotion": "Contempt",
        "description": "High arousal, negative valence.",
    },
    "Disgust": {
        "valence_direction": "negative",
        "hume_emotion": "Disgust",
        "description": "medium? arousal, negative valence.",
    },
    "Distress": {
        "valence_direction": "negative", 
        "hume_emotion": "Distress",
        "description": "high arousal, negative valence.",
    },
    "Fear": {
        "valence_direction": "negative",
        "hume_emotion": "Fear",
        "description": "any? arousal, negative valence.",
    },
    "Sadness": {
        "valence_direction": "negative",
        "hume_emotion": "Sadness",
        "description": "low arousal, negative valence.",
    },
    "Neutral": {
        "valence_direction": "zero", #should be as neutral as possible
        "hume_emotion": "Neutral",
        "description": "low arousal, neutral valence.",
    },
}

AVAILABLE_EMOTIONS = list(EMOTION_CONFIG.keys())


# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

.stApp { background-color: #0a0a0f; }
header[data-testid="stHeader"] { background: transparent; }
h1 { font-family: 'Space Mono', monospace !important; letter-spacing: -1px; }
h2, h3 { font-family: 'Space Mono', monospace !important; }

.metric-card {
    background: linear-gradient(135deg, #13131f 0%, #1a1a2e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.metric-card h4 { margin: 0 0 0.3rem 0; font-size: 0.75rem; color: #7878a8; text-transform: uppercase; letter-spacing: 2px; }
.metric-card .val { font-family: 'Space Mono', monospace; font-size: 1.6rem; color: #c8f; }
.metric-card .sub { font-size: 0.8rem; color: #5858a8; }

.winner-badge {
    display: inline-block;
    background: linear-gradient(90deg, #6633cc, #33aaff);
    color: white;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 10px;
    border-radius: 99px;
    margin-left: 8px;
    vertical-align: middle;
}

.score-breakdown-card {
    background: #13131f;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
}
.score-breakdown-card .label { color: #7878a8; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; }
.score-breakdown-card .bar-wrap { background: #1a1a2e; border-radius: 4px; height: 6px; margin: 4px 0 2px 0; }
.score-breakdown-card .bar-fill { border-radius: 4px; height: 6px; }

.stTabs [data-baseweb="tab-list"] { background: #13131f; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #7878a8; font-family: 'Space Mono', monospace; font-size: 0.8rem; }
.stTabs [aria-selected="true"] { background: #2a2a4a !important; color: #c8f !important; border-radius: 6px; }

[data-testid="stFileUploader"] {
    background: #13131f;
    border: 1px dashed #2a2a4a;
    border-radius: 12px;
    padding: 0.5rem;
}

[data-testid="stSidebar"] { background-color: #0d0d1a; border-right: 1px solid #1e1e3a; }

.stButton > button {
    background: linear-gradient(90deg, #6633cc, #3366ff);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    padding: 0.6rem 1.2rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.info-box {
    background: #13131f;
    border-left: 3px solid #6633cc;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #a0a0cc;
}

.emotion-box {
    background: #13131f;
    border-left: 3px solid #ff6644;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #a0a0cc;
}

hr { border-color: #1e1e3a; }
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

[data-testid="stStatusWidget"] p,
[data-testid="stStatusWidget"] label { color: #a0a0cc !important; }
[data-testid="stExpander"] p { color: #a0a0cc !important; }

[data-testid="stMarkdownContainer"] code {
    background-color: #1a1a2e !important;
    color: #cc44ff !important;
    border: 1px solid #2a2a4a;
    border-radius: 4px;
    padding: 1px 5px;
}

.rank-row {
    display: flex; align-items: center; gap: 12px;
    background: #13131f; border: 1px solid #1e1e3a;
    border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 0.4rem;
}
.rank-num { font-family: 'Space Mono', monospace; font-size: 1.1rem; color: #6633cc; width: 28px; }
.rank-name { font-weight: 600; flex: 1; }
.rank-score { font-family: 'Space Mono', monospace; font-size: 0.9rem; color: #33aaff; }
</style>
""", unsafe_allow_html=True)


# ── Lazy-load heavy deps ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading acoustic models…")
def load_smile():
    import opensmile
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    return smile


# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_egemaps(smile, audio_bytes, filename):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        feats = smile.process_file(tmp_path)
        return feats.iloc[0]
    finally:
        os.unlink(tmp_path)


@st.cache_resource(show_spinner="Loading emotion model…")
def load_arousal_model():
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features, **kwargs):
            x = self.dropout(features)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            return self.out_proj(x)

    class EmotionModel(Wav2Vec2PreTrainedModel):
        _tied_weights_keys = []
        all_tied_weights_keys = {}

        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.init_weights()

        def forward(self, input_values):
            hidden_states = self.wav2vec2(input_values).last_hidden_state
            hidden_states = torch.mean(hidden_states, dim=1)
            return self.classifier(hidden_states)

    model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def extract_arousal(processor_and_model, audio_bytes):
    import torch
    import soundfile as sf

    processor, model = processor_and_model
    buf = io.BytesIO(audio_bytes)
    try:
        data, sr = sf.read(buf)
    except Exception:
        return None

    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        target_len = int(len(data) * 16000 / sr)
        data = np.interp(np.linspace(0, len(data) - 1, target_len),
                         np.arange(len(data)), data)

    inputs = processor(data.astype(np.float32), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs["input_values"])

    scores = logits[0].tolist()
    return {"arousal": scores[0], "dominance": scores[1], "valence": scores[2]}


def compute_expressivity(feats):
    components = {
        "Pitch Range (F0 range)":     feats.get("F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2", np.nan),
        "Pitch Variability (F0 std)": feats.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", np.nan),
        "Loudness Variability":       feats.get("loudness_sma3_stddevNorm", np.nan),
        "Shimmer (local)":            feats.get("shimmerLocaldB_sma3nz_amean", np.nan),
        # "HNR (voice quality)":        feats.get("HNRdBACF_sma3nz_amean", np.nan),
    }
    return components


def run_hume_analysis(ready_files, results):
    """
    Runs Hume prosody analysis for all models.
    Returns dict: name → list of per-segment emotion dicts,
                  and name → averaged emotion scores dict.
    """
    hume_raw = {}      # name → list of per-segment dicts
    hume_means = {}    # name → {emotion: mean_score}
    hume_predictions = {}  # name → raw predictions object (for tab4 display)

    progress_text = st.empty()

    for name, file_obj in ready_files:
        audio_bytes = results[name]["audio"]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            progress_text.markdown(f'<div class="info-box">⏳ Submitting <strong>{name}</strong> to Hume...</div>', unsafe_allow_html=True)

            configs = InferenceBaseRequest(
                models=Models(prosody=Prosody(granularity="utterance"))
            )
            with open(tmp_path, "rb") as f:
                job_id = client.expression_measurement.batch.start_inference_job_from_local_file(
                    file=[f],
                    json=configs
                )

            while True:
                job_status = client.expression_measurement.batch.get_job_details(id=job_id)
                current_state = job_status.state.status
                if current_state == "COMPLETED":
                    break
                elif current_state == "FAILED":
                    st.warning(f"Hume job failed for {name}.")
                    break
                else:
                    time.sleep(3)

            predictions = client.expression_measurement.batch.get_job_predictions(job_id)
            hume_predictions[name] = predictions

            all_segment_emotions = []
            for result in predictions:
                group_predictions = result.results.predictions[0].models.prosody.grouped_predictions
                for group in group_predictions:
                    for prediction in group.predictions:
                        seg_dict = {e.name: e.score for e in prediction.emotions}
                        all_segment_emotions.append(seg_dict)

            hume_raw[name] = all_segment_emotions
            if all_segment_emotions:
                df_all = pd.DataFrame(all_segment_emotions)
                hume_means[name] = df_all.mean().to_dict()
            else:
                hume_means[name] = {}

        except Exception as e:
            st.warning(f"Hume failed for {name}: {e}")
            hume_raw[name] = []
            hume_means[name] = {}
        finally:
            os.unlink(tmp_path)

    progress_text.empty()
    return hume_raw, hume_means, hume_predictions


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE FUNCTION
# This is the central scoring function. Edit weights and logic here.
#
# Weights:
#   ACOUSTIC_WEIGHT  = 0.40  (eGeMAPS pitch, loudness, shimmer, HNR)
#   AROUSAL_WEIGHT   = 0.30  (wav2vec2 arousal dimension)
#   VALENCE_WEIGHT   = 0.15  (wav2vec2 valence, direction-aware per emotion)
#   HUME_WEIGHT      = 0.15  (Hume target-emotion detection score)
#
# All sub-scores are z-scored across models before weighting so they
# live on the same scale and no single component dominates by magnitude.
# ══════════════════════════════════════════════════════════════════════════════

ACOUSTIC_WEIGHT = 0.42
AROUSAL_WEIGHT  = 0.32
VALENCE_WEIGHT  = 0.10
HUME_WEIGHT     = 0.16


def compute_composite_score(
    model_list,
    df_comp,          # pd.DataFrame: models × acoustic components
    results,          # dict: name → {arousal: {arousal, valence, dominance}, ...}
    hume_means,       # dict: name → {emotion_name: mean_score}
    emotion_config,   # dict: from EMOTION_CONFIG[selected_emotion]
    use_arousal=True,
    use_hume=True,
):
    """
    Computes the composite expressivity score for each model.

    Returns:
        composite      pd.Series  — final weighted score (higher = better match)
        sub_scores     dict       — {component_name: pd.Series} for breakdown display
    """

    sub_scores = {}

    # ── 1. Acoustic sub-score (40%) ──────────────────────────────────────────
    # Invert HNR: breathy/dynamic voices are more expressive.
    df_norm = df_comp.copy()
    # df_norm["HNR (voice quality)"] = -df_norm["HNR (voice quality)"]
    df_zscored = (df_norm - df_norm.mean()) / (df_norm.std() + 1e-8)
    acoustic_score = df_zscored.mean(axis=1)  # average across features
    sub_scores["acoustic"] = acoustic_score

    # ── 2. Arousal sub-score (30%) ───────────────────────────────────────────
    if use_arousal:
        arousal_raw = pd.Series({
            name: (results[name]["arousal"] or {}).get("arousal", 0.0)
            for name in model_list
        })
        arousal_z = (arousal_raw - arousal_raw.mean()) / (arousal_raw.std() + 1e-8)
    else:
        arousal_z = pd.Series({name: 0.0 for name in model_list})
    sub_scores["arousal"] = arousal_z

    # ── 3. Valence sub-score (15%) ───────────────────────────────────────────
    # Direction is set by the selected emotion's config.
    # "negative" target → lower raw valence = better → we negate before z-scoring.
    # "positive" target → higher raw valence = better → keep as-is.
    if use_arousal:
        valence_raw = pd.Series({
            name: (results[name]["arousal"] or {}).get("valence", 0.5)
            for name in model_list
        })
        direction = emotion_config.get("valence_direction", "positive")
        if direction == "negative":
            valence_signed = -valence_raw   # flip: low valence → high score
        else:
            valence_signed = valence_raw

        valence_z = (valence_signed - valence_signed.mean()) / (valence_signed.std() + 1e-8)
    else:
        valence_z = pd.Series({name: 0.0 for name in model_list})
    sub_scores["valence"] = valence_z

    # ── 4. Hume sub-score (15%) ──────────────────────────────────────────────
    # Score = how strongly Hume detected the target emotion (mean score across utterances).
    # We additionally compute a rank bonus: if the target emotion is in Hume's top-3
    # emotions for a model, we add a small bonus (0.25 z-unit equivalent) to reward
    # models where the emotion is clearly the dominant affect — not just detectable.
    if use_hume and hume_means:
        target_emotion = emotion_config.get("hume_emotion", "")

        # ── Rank-weighted mean (Option 1) ────────────────────────────────────
        # For each model, score = target_emotion_score / rank_position (1-indexed).
        # This rewards both high raw confidence AND high salience (low rank index)
        # simultaneously, with a smooth gradient rather than discrete bonus tiers.
        # The result is then normalized across models so it sums to 1 before
        # z-scoring, keeping the scale stable regardless of how many emotions
        # Hume returns.
        #
        # Formula per model:
        #   raw_weighted = score(target) / (rank(target) + 1)
        #   normalized   = raw_weighted / Σ(score_i / (rank_i + 1))  for all emotions
        #
        # Using (rank + 1) rather than rank avoids division-by-zero for rank=0
        # and smoothly compresses the weight of lower-ranked emotions.
        # ─────────────────────────────────────────────────────────────────────
        hume_raw_scores = {}
        for name in model_list:
            emotions = hume_means.get(name, {})
            if not emotions:
                hume_raw_scores[name] = 0.0
                continue

            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

            # Build the full rank-weighted sum across ALL emotions (normalization denominator)
            total_rank_weighted = sum(
                score / (rank + 1)
                for rank, (_, score) in enumerate(sorted_emotions)
            )

            # Find rank of the target emotion
            target_rank = next(
                (i for i, (k, _) in enumerate(sorted_emotions) if k == target_emotion),
                None
            )

            if target_rank is not None:
                target_score = emotions[target_emotion]
                raw_weighted = target_score / (target_rank + 1)
                # Normalize: ratio of target's rank-weighted score to the total
                hume_raw_scores[name] = raw_weighted / (total_rank_weighted + 1e-8)
            else:
                # Target emotion not detected at all
                hume_raw_scores[name] = 0.0

        hume_series = pd.Series(hume_raw_scores)
        hume_z = (hume_series - hume_series.mean()) / (hume_series.std() + 1e-8)
    else:
        hume_z = pd.Series({name: 0.0 for name in model_list})
    sub_scores["hume"] = hume_z

    # ── Weighted combination ─────────────────────────────────────────────────
    # Normalize weights in case arousal/hume are disabled.
    w_acoustic = ACOUSTIC_WEIGHT
    w_arousal  = AROUSAL_WEIGHT  if use_arousal else 0.0
    w_valence  = VALENCE_WEIGHT  if use_arousal else 0.0
    w_hume     = HUME_WEIGHT     if use_hume    else 0.0

    total_w = w_acoustic + w_arousal + w_valence + w_hume
    if total_w == 0:
        total_w = 1.0  # guard against division by zero

    composite = (
        w_acoustic / total_w * acoustic_score
        + w_arousal  / total_w * arousal_z
        + w_valence  / total_w * valence_z
        + w_hume     / total_w * hume_z
    )

    return composite.sort_values(ascending=False), sub_scores


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════
PLOTLY_THEME = dict(
    paper_bgcolor="#0a0a0f",
    plot_bgcolor="#13131f",
    font=dict(family="DM Sans", color="#a0a0cc"),
    xaxis=dict(gridcolor="#1e1e3a", zerolinecolor="#1e1e3a"),
    yaxis=dict(gridcolor="#1e1e3a", zerolinecolor="#1e1e3a"),
)

ACCENT_COLORS = ["#cc44ff", "#33aaff", "#ff6644", "#44ffcc", "#ffcc33", "#ff44aa"]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Expressivity Analysis")
    st.markdown('<div class="info-box">Upload one <strong>.wav</strong> file per TTS model. All clips should read the <strong>same script</strong>.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Emotion target selection ──────────────────────────────────────────────
    st.markdown("### 🎯 Target Emotion")
    selected_emotion = st.selectbox(
        "What emotion should the TTS convey?",
        options=AVAILABLE_EMOTIONS,
        index=0,
        help="This affects valence scoring (which direction = better) and Hume scoring (which emotion to reward)."
    )
    emotion_cfg = EMOTION_CONFIG[selected_emotion]
    valence_dir_label = "← negative (low valence)" if emotion_cfg["valence_direction"] == "negative" else "→ positive (high valence)"
    st.markdown(f'<div class="emotion-box"><strong>{selected_emotion}</strong><br>'
                f'Valence target: {valence_dir_label}<br>'
                f'Hume looks for: <code>{emotion_cfg["hume_emotion"]}</code><br>'
                f'<span style="font-size:0.78rem">{emotion_cfg["description"]}</span></div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Models to compare")

    num_models = st.number_input("How many TTS models?", min_value=2, max_value=20, value=3, step=1)

    model_names = []
    uploaded_files = []
    for i in range(num_models):
        st.markdown(f"**Model {i+1}**")
        name = st.text_input(f"Name", value=f"TTS Model {i+1}", key=f"name_{i}", label_visibility="collapsed")
        f = st.file_uploader(f"Upload WAV", type=["wav", "mp3", "flac"], key=f"file_{i}", label_visibility="collapsed")
        model_names.append(name)
        uploaded_files.append(f)

    st.markdown("---")
    use_arousal = st.toggle("🧠 Include Arousal/Valence Model", value=True,
                            help="Adds audeering wav2vec2 arousal/valence. Slower but richer.")
    use_hume = st.toggle("🦨 Include Hume Analysis", value=True,
                         help="Runs Hume prosody API. Requires API key. Adds ~30s per model.")

    st.markdown("---")
    st.markdown("### Score Weights")
    actual_w_acoustic = ACOUSTIC_WEIGHT
    actual_w_arousal  = AROUSAL_WEIGHT if use_arousal else 0.0
    actual_w_valence  = VALENCE_WEIGHT if use_arousal else 0.0
    actual_w_hume     = HUME_WEIGHT    if use_hume    else 0.0
    total_w_display   = actual_w_acoustic + actual_w_arousal + actual_w_valence + actual_w_hume or 1.0

    def pct(w): return f"{w / total_w_display * 100:.0f}%"

    st.markdown(f"""
    | Component | Weight |
    |-----------|--------|
    | Acoustic  | {pct(actual_w_acoustic)} |
    | Arousal   | {pct(actual_w_arousal)} |
    | Valence   | {pct(actual_w_valence)} |
    | Hume      | {pct(actual_w_hume)} |
    """)

    analyze = st.button("▶ Run Analysis", use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="info-box" style="font-size:0.75rem">Acoustic: <strong>eGeMAPS v02</strong> via OpenSmile<br>Emotion dims: <strong>audeering/wav2vec2-large-robust</strong><br>Emotion detection: <strong>Hume AI Prosody</strong></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# TTS Expressivity Analysis")
st.markdown(f"*Target emotion: **{selected_emotion}** · Purely acoustic/prosodic — no text or semantics used*")
st.markdown("---")

ready_files = [(model_names[i], uploaded_files[i]) for i in range(num_models) if uploaded_files[i] is not None]

if not analyze:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="metric-card">
            <h4>What this measures</h4>
            <div style="font-size:0.85rem;color:#a0a0cc;margin-top:0.5rem">
            Pitch range & variability<br>Loudness dynamics<br>Voice quality (shimmer, HNR)<br>
            Arousal & valence scores<br>Hume emotion detection
            </div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <h4>Score breakdown</h4>
            <div style="font-size:0.85rem;color:#a0a0cc;margin-top:0.5rem">
            40% Acoustic features<br>30% Arousal (energy/activation)<br>
            15% Valence (emotion polarity)<br>15% Hume emotion detection
            </div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <h4>How to use</h4>
            <div style="font-size:0.85rem;color:#a0a0cc;margin-top:0.5rem">
            1. Pick your target emotion<br>2. Upload one WAV per TTS model<br>3. Click ▶ Run Analysis
            </div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="info-box">👈 Configure your target emotion and models in the sidebar, then click <strong>Run Analysis</strong>.</div>', unsafe_allow_html=True)
    st.stop()

if len(ready_files) < 2:
    st.warning("Please upload at least 2 audio files to compare.")
    st.stop()

# ── Run analysis ───────────────────────────────────────────────────────────────
smile = load_smile()
arousal_pipe = load_arousal_model() if use_arousal else None

results = {}

progress = st.progress(0, text="Extracting acoustic features…")
for idx, (name, file) in enumerate(ready_files):
    audio_bytes = file.read()
    progress.progress(idx / len(ready_files), text=f"Processing {name}…")

    feats = extract_egemaps(smile, audio_bytes, file.name)
    components = compute_expressivity(feats)

    arousal_out = None
    if use_arousal and arousal_pipe:
        arousal_out = extract_arousal(arousal_pipe, audio_bytes)

    results[name] = {
        "feats": feats,
        "components": components,
        "arousal": arousal_out,
        "audio": audio_bytes,
    }

progress.progress(1.0, text="Acoustic features done!")
progress.empty()

# ── Hume analysis (runs during main analysis, not lazily in tab) ───────────────
hume_raw = {}
hume_means = {}
hume_predictions = {}

if use_hume:
    with st.status("Running Hume AI analysis…", expanded=False) as hume_status:
        hume_raw, hume_means, hume_predictions = run_hume_analysis(ready_files, results)
        hume_status.update(label="Hume analysis complete!", state="complete")

model_list = list(results.keys())
colors = {name: ACCENT_COLORS[i % len(ACCENT_COLORS)] for i, name in enumerate(model_list)}

# ── eGeMAPS matrix ─────────────────────────────────────────────────────────────
comp_matrix = {name: results[name]["components"] for name in model_list}
df_comp = pd.DataFrame(comp_matrix).T

# ── Composite score ────────────────────────────────────────────────────────────
composite, sub_scores = compute_composite_score(
    model_list=model_list,
    df_comp=df_comp,
    results=results,
    hume_means=hume_means,
    emotion_config=emotion_cfg,
    use_arousal=use_arousal,
    use_hume=use_hume,
)

winner = composite.index[0]

# z-scored acoustic matrix for radar/feature charts
df_norm = df_comp.copy()
# df_norm["HNR (voice quality)"] = -df_norm["HNR (voice quality)"]
df_zscored = (df_norm - df_norm.mean()) / (df_norm.std() + 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Rankings", "📊 Acoustic Features", "🧠 Arousal & Emotion", "🦨 Hume Emotion", "🔬 Raw Data"])


# ── TAB 1: Rankings ────────────────────────────────────────────────────────────
with tab1:
    st.markdown(f"### Overall Expressivity Ranking — *{selected_emotion}*")

    enabled_parts = ["acoustic"]
    if use_arousal:
        enabled_parts += ["arousal", "valence"]
    if use_hume:
        enabled_parts += ["hume"]
    st.markdown(f"*Composite: {' + '.join(enabled_parts)}*")

    col_rank, col_radar = st.columns([1, 1.4])

    with col_rank:
        for rank, (name, score) in enumerate(composite.items()):
            badge = '<span class="winner-badge">WINNER</span>' if rank == 0 else ""
            st.markdown(f"""<div class="rank-row">
                <div class="rank-num">#{rank+1}</div>
                <div class="rank-name">{name}{badge}</div>
                <div class="rank-score">{score:+.3f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="info-box">Score is z-scored composite — relative within this session. Higher = better match for target emotion.</div>', unsafe_allow_html=True)

    with col_radar:
        cats = list(df_zscored.columns)
        fig_radar = go.Figure()
        for name in model_list:
            vals = df_zscored.loc[name].tolist()
            vals += [vals[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=cats + [cats[0]],
                name=name,
                line=dict(color=colors[name], width=2),
                fill="toself",
                fillcolor=colors[name],
                opacity=0.15,
            ))
        fig_radar.update_layout(
            **PLOTLY_THEME,
            polar=dict(
                bgcolor="#13131f",
                radialaxis=dict(visible=True, gridcolor="#2a2a4a", color="#5858a8"),
                angularaxis=dict(gridcolor="#2a2a4a", color="#7878a8"),
            ),
            legend=dict(font=dict(color="#a0a0cc")),
            margin=dict(l=40, r=40, t=40, b=40),
            height=380,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Sub-score breakdown ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Score Component Breakdown")
    st.markdown("*All sub-scores are z-scored across models before weighting.*")

    score_labels = {
        "acoustic": f"Acoustic Features ({pct(actual_w_acoustic)})",
        "arousal":  f"Arousal ({pct(actual_w_arousal)})",
        "valence":  f"Valence — {emotion_cfg['valence_direction']} target ({pct(actual_w_valence)})",
        "hume":     f"Hume '{emotion_cfg['hume_emotion']}' detection ({pct(actual_w_hume)})",
    }
    score_colors = {
        "acoustic": "#cc44ff",
        "arousal":  "#33aaff",
        "valence":  "#ff6644",
        "hume":     "#44ffcc",
    }

    fig_breakdown = go.Figure()
    for component, label in score_labels.items():
        series = sub_scores[component]
        fig_breakdown.add_trace(go.Bar(
            name=label,
            x=model_list,
            y=[series.get(name, 0) for name in model_list],
            marker_color=score_colors[component],
            opacity=0.85,
        ))
    fig_breakdown.add_hline(y=0, line_color="#3a3a5a", line_width=1)
    fig_breakdown.update_layout(
        **PLOTLY_THEME,
        barmode="group",
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(font=dict(color="#a0a0cc"), orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)

    st.markdown("---")
    st.markdown("### Listen & Compare")
    cols = st.columns(len(model_list))
    for i, name in enumerate(model_list):
        with cols[i]:
            st.markdown(f"**{name}**")
            st.audio(results[name]["audio"])


# ── TAB 2: Acoustic Features ───────────────────────────────────────────────────
with tab2:
    st.markdown("### eGeMAPS Acoustic Feature Breakdown")

    feature_labels = list(df_comp.columns)
    fig_bar = go.Figure()
    for name in model_list:
        fig_bar.add_trace(go.Bar(
            name=name,
            x=feature_labels,
            y=df_comp.loc[name].tolist(),
            marker_color=colors[name],
            opacity=0.85,
        ))
    fig_bar.update_layout(
        **PLOTLY_THEME,
        barmode="group",
        legend=dict(font=dict(color="#a0a0cc")),
        margin=dict(l=20, r=20, t=20, b=80),
        height=380,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Feature Explanations")
    feat_info = {
        "Pitch Range (F0 range)":     ("Semitones between lowest & highest pitch", "Higher = wider melodic range → more expressive"),
        "Pitch Variability (F0 std)": ("Normalized std dev of pitch", "Higher = more dynamic intonation"),
        "Loudness Variability":       ("Normalized std dev of loudness (sone)", "Higher = more dramatic volume changes"),
        "Shimmer (local)":            ("dB variation between consecutive glottal cycles", "Some shimmer = natural, expressive voice"),
        "HNR (voice quality)":        ("Harmonics-to-noise ratio in dB", "Lower HNR here = breathy, more expressive (inverted in composite)"),
    }
    c1, c2 = st.columns(2)
    for i, (feat, (what, why)) in enumerate(feat_info.items()):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""<div class="metric-card">
                <h4>{feat}</h4>
                <div style="font-size:0.82rem;color:#7878a8">{what}</div>
                <div style="font-size:0.82rem;color:#a0a0cc;margin-top:0.3rem">→ {why}</div>
            </div>""", unsafe_allow_html=True)


# ── TAB 3: Arousal & Emotion ───────────────────────────────────────────────────
with tab3:
    if not use_arousal:
        st.info("Enable the Arousal/Valence Model toggle in the sidebar to see this section.")
    else:
        st.markdown("### Arousal · Valence · Dominance")
        st.markdown("*From `audeering/wav2vec2-large-robust` — audio-only, no text used.*")

        dims = ["arousal", "valence", "dominance"]
        avd_data = {}
        for name in model_list:
            ar = results[name]["arousal"] or {}
            avd_data[name] = {d: ar.get(d, 0) for d in dims}
        df_avd = pd.DataFrame(avd_data).T

        # Valence scoring explanation
        direction = emotion_cfg["valence_direction"]
        direction_label = "Lower valence = better (negative emotion target)" if direction == "negative" \
                          else "Higher valence = better (positive emotion target)"
        st.markdown(f'<div class="emotion-box">Valence scoring for <strong>{selected_emotion}</strong>: {direction_label}</div>', unsafe_allow_html=True)

        st.markdown("#### Arousal vs Valence (Russell's Circumplex)")

        # Draw shaded target zone based on emotion direction
        fig_scatter = go.Figure()

        # Shade the target quadrant
        if direction == "negative":
            # Left half = negative valence target
            fig_scatter.add_shape(type="rect", x0=0, x1=0.5, y0=0, y1=1,
                                  fillcolor="rgba(255, 100, 68, 0.06)", line_width=0)
            target_label = "Target zone (negative valence)"
        else:
            fig_scatter.add_shape(type="rect", x0=0.5, x1=1, y0=0, y1=1,
                                  fillcolor="rgba(51, 170, 255, 0.06)", line_width=0)
            target_label = "Target zone (positive valence)"

        for name in model_list:
            fig_scatter.add_trace(go.Scatter(
                x=[df_avd.loc[name, "valence"]],
                y=[df_avd.loc[name, "arousal"]],
                mode="markers+text",
                name=name,
                text=[name],
                textposition="top center",
                marker=dict(size=18, color=colors[name], symbol="circle",
                            line=dict(width=2, color="white")),
            ))

        fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="#2a2a4a")
        fig_scatter.add_vline(x=0.5, line_dash="dash", line_color="#2a2a4a")
        for label, x, y in [
            ("Excited", 0.75, 0.82), ("Calm", 0.75, 0.18),
            ("Anxious/Angry", 0.25, 0.82), ("Sad", 0.25, 0.18)
        ]:
            fig_scatter.add_annotation(x=x, y=y, text=label, showarrow=False,
                                       font=dict(color="#3a3a6a", size=11))

        fig_scatter.update_layout(
            **PLOTLY_THEME, height=400, showlegend=True,
            margin=dict(l=40, r=40, t=20, b=40),
            legend=dict(font=dict(color="#a0a0cc")),
        )
        fig_scatter.update_xaxes(title="Valence →", range=[0, 1])
        fig_scatter.update_yaxes(title="Arousal →", range=[0, 1], gridcolor="#1e1e3a")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("#### All Three Dimensions")
        fig_avd = go.Figure()
        for dim in dims:
            fig_avd.add_trace(go.Bar(
                name=dim.capitalize(),
                x=model_list,
                y=[df_avd.loc[n, dim] for n in model_list],
                opacity=0.85,
            ))
        fig_avd.update_layout(
            **PLOTLY_THEME, barmode="group", height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(font=dict(color="#a0a0cc")),
        )
        st.plotly_chart(fig_avd, use_container_width=True)

        st.markdown('<div class="info-box">Valence score for this session is direction-aware: the composite rewards models whose valence is on the correct side for the target emotion.</div>', unsafe_allow_html=True)


# ── TAB 4: Hume ───────────────────────────────────────────────────────────────
with tab4:
    if not use_hume:
        st.info("Enable the Hume Analysis toggle in the sidebar to see this section.")
    elif not hume_predictions:
        st.warning("Hume analysis produced no results. Check your API key and audio files.")
    else:
        st.markdown(f"### Hume Prosody Analysis — Target: *{selected_emotion}*")
        st.markdown("Prosody-based emotion analysis from Hume AI — no text used.")

        target_emo = emotion_cfg["hume_emotion"]

        # Summary comparison across models
        st.markdown("#### Target Emotion Score Comparison")
        target_scores = {name: hume_means.get(name, {}).get(target_emo, 0.0) for name in model_list}
        fig_target = go.Figure(go.Bar(
            x=list(target_scores.keys()),
            y=list(target_scores.values()),
            marker_color=[colors[n] for n in target_scores.keys()],
            opacity=0.85,
            text=[f"{v:.2%}" for v in target_scores.values()],
            textposition="outside",
        ))
        fig_target.update_layout(
            **PLOTLY_THEME, height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title=f"Mean '{target_emo}' score",
        )
        st.plotly_chart(fig_target, use_container_width=True)

        st.markdown("---")

        # Per-model detail
        for name in model_list:
            if name not in hume_predictions or not hume_means.get(name):
                continue

            st.subheader(f"🎙️ {name}")
            emotions = hume_means[name]
            global_sorted = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

            # Find rank of target emotion
            target_rank = next((i for i, (k, _) in enumerate(global_sorted) if k == target_emo), -1)
            rank_label = f"#{target_rank + 1}" if target_rank >= 0 else "Not detected"

            m1, m2, m3 = st.columns(3)
            m1.metric(label=f"'{target_emo}' score", value=f"{emotions.get(target_emo, 0):.2%}")
            m2.metric(label=f"'{target_emo}' rank", value=rank_label)
            m3.metric(label="Top emotion", value=global_sorted[0][0] if global_sorted else "—",
                      delta=f"{global_sorted[0][1]:.2%}" if global_sorted else None)

            top10 = global_sorted[:10]
            emo_names = [e[0] for e in top10]
            emo_vals  = [e[1] for e in top10]
            bar_colors = ["#ff6644" if e == target_emo else colors.get(name, "#cc44ff") for e in emo_names]

            fig_hume = go.Figure(go.Bar(
                x=emo_names,
                y=emo_vals,
                marker_color=bar_colors,
                opacity=0.85,
                text=[f"{v:.2%}" for v in emo_vals],
                textposition="outside",
            ))
            fig_hume.update_layout(
                **PLOTLY_THEME, height=300,
                margin=dict(l=20, r=20, t=20, b=60),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_hume, use_container_width=True)
            st.markdown("---")


# ── TAB 5: Raw Data ────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### Raw eGeMAPS Features")
    selected = st.selectbox("Select model to inspect", model_list, key="raw_data_model_select")
    feats_df = results[selected]["feats"].to_frame(name="Value")
    feats_df.index.name = "Feature"
    st.dataframe(feats_df.style.format("{:.5f}"), use_container_width=True, height=400)

    all_feats = pd.DataFrame({n: results[n]["feats"] for n in model_list}).T
    all_feats.index.name = "Model"
    csv = all_feats.to_csv()
    st.download_button("📥 Download All Features (CSV)", csv, "tts_egemaps_features.csv", "text/csv")

    if use_arousal:
        st.markdown("### Arousal / Valence / Dominance Raw Scores")
        avd_rows = []
        for name in model_list:
            ar = results[name]["arousal"] or {}
            avd_rows.append({"Model": name, **ar})
        st.dataframe(pd.DataFrame(avd_rows).set_index("Model"), use_container_width=True)

    if use_hume and hume_means:
        st.markdown("### Hume Mean Emotion Scores")
        hume_df = pd.DataFrame(hume_means).T
        hume_df.index.name = "Model"
        st.dataframe(hume_df.style.format("{:.4f}"), use_container_width=True, height=400)
        hume_csv = hume_df.to_csv()
        st.download_button("📥 Download Hume Scores (CSV)", hume_csv, "tts_hume_scores.csv", "text/csv")

    st.markdown("### Composite Score Sub-Scores")
    sub_df = pd.DataFrame({k: v for k, v in sub_scores.items()})
    sub_df.index.name = "Model"
    sub_df["COMPOSITE"] = composite
    st.dataframe(sub_df.style.format("{:.4f}"), use_container_width=True)