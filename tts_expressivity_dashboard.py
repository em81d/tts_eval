import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile, os, io
from dotenv import load_dotenv


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TTS Expressivity Lab",
    page_icon="🎙️",
    layout="wide",
)


load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# EMOTION CONFIGURATION
# Maps emotion label → scoring parameters for valence.
# valence_direction: "negative" → lower valence scores better
#                   "positive" → higher valence scores better
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


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE FUNCTION
# This is the central scoring function. Edit weights and logic here.
#
# Weights:
#   ACOUSTIC_WEIGHT  = 0.42  (eGeMAPS pitch, loudness, shimmer, HNR)
#   AROUSAL_WEIGHT   = 0.32  (wav2vec2 arousal dimension)
#   VALENCE_WEIGHT   = 0.10  (wav2vec2 valence, direction-aware per emotion)
#
# All sub-scores are z-scored across models before weighting so they
# live on the same scale and no single component dominates by magnitude.
# ══════════════════════════════════════════════════════════════════════════════

ACOUSTIC_WEIGHT = 0.42
AROUSAL_WEIGHT  = 0.32
VALENCE_WEIGHT  = 0.10


def compute_composite_score(
    model_list,
    df_comp,          # pd.DataFrame: models × acoustic components
    results,          # dict: name → {arousal: {arousal, valence, dominance}, ...}
    emotion_config,   # dict: from EMOTION_CONFIG[selected_emotion]
    use_arousal=True,
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

    # ── Weighted combination ─────────────────────────────────────────────────
    # Normalize weights in case arousal is disabled.
    w_acoustic = ACOUSTIC_WEIGHT
    w_arousal  = AROUSAL_WEIGHT  if use_arousal else 0.0
    w_valence  = VALENCE_WEIGHT  if use_arousal else 0.0

    total_w = w_acoustic + w_arousal + w_valence
    if total_w == 0:
        total_w = 1.0  # guard against division by zero

    composite = (
        w_acoustic / total_w * acoustic_score
        + w_arousal  / total_w * arousal_z
        + w_valence  / total_w * valence_z
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
        help="This affects valence scoring (which direction = better)."
    )
    emotion_cfg = EMOTION_CONFIG[selected_emotion]
    valence_dir_label = "← negative (low valence)" if emotion_cfg["valence_direction"] == "negative" else "→ positive (high valence)"
    st.markdown(f'<div class="emotion-box"><strong>{selected_emotion}</strong><br>'
                f'Valence target: {valence_dir_label}<br>'
                f'<span style="font-size:0.78rem">{emotion_cfg["description"]}</span></div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Upload Audio Files")
    uploaded_files = st.file_uploader("Upload WAV files (up to 45)", type=["wav", "mp3", "flac"], accept_multiple_files=True, key="folder_upload")

    model_names = [os.path.splitext(f.name)[0] for f in uploaded_files if f is not None]

    if uploaded_files:
        st.markdown("#### Customize Colors")
        # Initialize custom_colors in session state if not already present
        if "custom_colors" not in st.session_state:
            st.session_state.custom_colors = {}

        for i, name in enumerate(model_names):
            # Get current color from ACCENT_COLORS or session state
            default_color = ACCENT_COLORS[i % len(ACCENT_COLORS)]
            current_color = st.session_state.custom_colors.get(name, default_color)

            chosen_color = st.color_picker(f"Color for {name}", value=current_color, key=f"color_{name}")
            if chosen_color != current_color:
                st.session_state.custom_colors[name] = chosen_color

    st.markdown("---")

    use_arousal = st.toggle("🧠 Include Arousal/Valence Model", value=True,
                            help="Adds audeering wav2vec2 arousal/valence. Slower but richer.")

    st.markdown("---")
    st.markdown("### Score Weights")
    actual_w_acoustic = ACOUSTIC_WEIGHT
    actual_w_arousal  = AROUSAL_WEIGHT if use_arousal else 0.0
    actual_w_valence  = VALENCE_WEIGHT if use_arousal else 0.0
    total_w_display   = actual_w_acoustic + actual_w_arousal + actual_w_valence or 1.0

    def pct(w): return f"{w / total_w_display * 100:.0f}%"

    st.markdown(f"""
    | Component | Weight |
    |-----------|--------|
    | Acoustic  | {pct(actual_w_acoustic)} |
    | Arousal   | {pct(actual_w_arousal)} |
    | Valence   | {pct(actual_w_valence)} |
    """)

    analyze = st.button("▶ Run Analysis", use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="info-box" style="font-size:0.75rem">Acoustic: <strong>eGeMAPS v02</strong> via OpenSmile<br>Emotion dims: <strong>audeering/wav2vec2-large-robust</strong></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# TTS Expressivity Analysis")
st.markdown(f"*Target emotion: **{selected_emotion}** · Purely acoustic/prosodic — no text or semantics used*")
st.markdown("---")

# Prepare files for analysis
ready_files = []
for uploaded_file in uploaded_files:
    # Use the filename (without extension) as the model name
    name = os.path.splitext(uploaded_file.name)[0]
    ready_files.append((name, uploaded_file))

    if not analyze:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""<div class="metric-card">
                <h4>What this measures</h4>
                <div style="font-size:0.85rem;color:#a0a0cc;margin-top:0.5rem">
                Pitch range & variability<br>Loudness dynamics<br>Voice quality (shimmer, HNR)<br>
                Arousal & valence scores
                </div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="metric-card">
                <h4>Score breakdown</h4>
                <div style="font-size:0.85rem;color:#a0a0cc;margin-top:0.5rem">
                50% Acoustic features<br>38% Arousal (energy/activation)<br>
                12% Valence (emotion polarity)
                </div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class="metric-card">
                <h4>How to use</h4>
                <div style="font-size:0.85rem;color:#a0a0cc;margin-top:0.5rem">
                1. Pick your target emotion<br>2. Upload WAV files from a folder<br>3. Click ▶ Run Analysis
                </div></div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div class=\"info-box\">👈 Configure your target emotion and upload models in the sidebar, then click <strong>Run Analysis</strong>.</div>", unsafe_allow_html=True)
        st.stop()

if len(ready_files) < 2:
    st.warning("Please upload at least 2 audio files to compare. Sample files can be found in /resources.")
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

model_list = list(results.keys())
# Use custom colors from session state if available, otherwise use default ACCENT_COLORS
colors = {name: st.session_state.custom_colors.get(name, ACCENT_COLORS[i % len(ACCENT_COLORS)])
          for i, name in enumerate(model_list)}

# ── eGeMAPS matrix ─────────────────────────────────────────────────────────────
comp_matrix = {name: results[name]["components"] for name in model_list}
df_comp = pd.DataFrame(comp_matrix).T

# ── Composite score ────────────────────────────────────────────────────────────
composite, sub_scores = compute_composite_score(
    model_list=model_list,
    df_comp=df_comp,
    results=results,
    emotion_config=emotion_cfg,
    use_arousal=use_arousal,
)

winner = composite.index[0]

# z-scored acoustic matrix for radar/feature charts
df_norm = df_comp.copy()
# df_norm["HNR (voice quality)"] = -df_norm["HNR (voice quality)"]
df_zscored = (df_norm - df_norm.mean()) / (df_norm.std() + 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab5 = st.tabs(["🏆 Rankings", "📊 Acoustic Features", "🧠 Arousal & Emotion", "🔬 Raw Data"])


# ── TAB 1: Rankings ────────────────────────────────────────────────────────────
with tab1:
    st.markdown(f"### Overall Expressivity Ranking — *{selected_emotion}*")

    enabled_parts = ["acoustic"]
    if use_arousal:
        enabled_parts += ["arousal", "valence"]
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
    }
    score_colors = {
        "acoustic": "#cc44ff",
        "arousal":  "#33aaff",
        "valence":  "#ff6644",
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

    st.markdown("### Composite Score Sub-Scores")
    sub_df = pd.DataFrame({k: v for k, v in sub_scores.items()})
    sub_df.index.name = "Model"
    sub_df["COMPOSITE"] = composite
    st.dataframe(sub_df.style.format("{:.4f}"), use_container_width=True)