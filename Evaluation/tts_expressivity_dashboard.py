import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile, os, io
from dotenv import load_dotenv
import time




from hume import HumeClient
load_dotenv()
HUME_API_KEY = os.getenv("HUME_API_KEY")

from hume.expression_measurement.batch.types import Models, Prosody, InferenceBaseRequest


# Initialize the Hume Client
if not HUME_API_KEY:
    st.error("Missing HUME_API_KEY in .env file!")
    st.stop()

client = HumeClient(api_key=HUME_API_KEY)



# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TTS Expressivity Lab",
    page_icon="🎙️",
    layout="wide",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}
            

/* Main background */
.stApp { background-color: #0a0a0f; }

/* Hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* Title styling */
h1 { font-family: 'Space Mono', monospace !important; letter-spacing: -1px; }
h2, h3 { font-family: 'Space Mono', monospace !important; }

/* Cards */
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

/* Winner badge */
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

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #13131f; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #7878a8; font-family: 'Space Mono', monospace; font-size: 0.8rem; }
.stTabs [aria-selected="true"] { background: #2a2a4a !important; color: #c8f !important; border-radius: 6px; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #13131f;
    border: 1px dashed #2a2a4a;
    border-radius: 12px;
    padding: 0.5rem;
}

/* Sidebar */
[data-testid="stSidebar"] { background-color: #0d0d1a; border-right: 1px solid #1e1e3a; }

/* Buttons */
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

/* Info boxes */
.info-box {
    background: #13131f;
    border-left: 3px solid #6633cc;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #a0a0cc;
}

/* Divider */
hr { border-color: #1e1e3a; }

/* Plotly chart containers */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }


/* st.status container text */
[data-testid="stStatusWidget"] p,
[data-testid="stStatusWidget"] label {
    color: #a0a0cc !important;
}

/* st.write text inside the status expander */
[data-testid="stExpander"] p {
    color: #a0a0cc !important;
}
            
/* Inline code spans (the job ID backtick text) */
[data-testid="stMarkdownContainer"] code {
    background-color: #1a1a2e !important;
    color: #cc44ff !important;
    border: 1px solid #2a2a4a;
    border-radius: 4px;
    padding: 1px 5px;
}

            

/* Rank table */
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
        print("column names: " + str([col for col in feats.columns if "F0" in col]))
        return feats.iloc[0]
    finally:
        os.unlink(tmp_path)


#the arousal model from huggingface uses a weird setup, I am trying to get it to work
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
            hidden_states = torch.mean(hidden_states, dim=1)  # pool across time
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
        logits = model(inputs["input_values"])  # shape: [1, 3]

    scores = logits[0].tolist()
    return {"arousal": scores[0], "dominance": scores[1], "valence": scores[2]}



def compute_expressivity(feats):
    """
    Compute a composite expressivity score from eGeMAPS features.
    Higher = more expressive.
    Components:
      - F0 range (pitch range)
      - F0 std dev (pitch variability)
      - Loudness std dev
      - Speaking rate proxy (voiced segments ratio)
      - Spectral flux / shimmer
    All are z-scored within the session for fair comparison.
    """
    components = {
        "Pitch Range (F0 range)":    feats.get("F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2", np.nan),
        "Pitch Variability (F0 std)": feats.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", np.nan),
        "Loudness Variability":       feats.get("loudness_sma3_stddevNorm", np.nan),
        "Shimmer (local)":            feats.get("shimmerLocaldB_sma3nz_amean", np.nan),
        "HNR (voice quality)":        feats.get("HNRdBACF_sma3nz_amean", np.nan),
    }
    return components


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
    st.markdown("### Models to compare")

    num_models = st.number_input("How many TTS models?", min_value=2, max_value=6, value=3, step=1)

    model_names = []
    uploaded_files = []
    for i in range(num_models):
        st.markdown(f"**Model {i+1}**")
        name = st.text_input(f"Name", value=f"TTS Model {i+1}", key=f"name_{i}", label_visibility="collapsed")
        f = st.file_uploader(f"Upload WAV", type=["wav", "mp3", "flac"], key=f"file_{i}", label_visibility="collapsed")
        model_names.append(name)
        uploaded_files.append(f)

    st.markdown("---")
    use_arousal = st.toggle("🧠 Include Arousal Model", value=True,
                            help="Adds audeering wav2vec2 arousal/valence. Slower but richer.")
    
    analyze = st.button("▶ Run Analysis", width='stretch')

    st.markdown("---")
    st.markdown('<div class="info-box" style="font-size:0.75rem">Acoustic features: <strong>eGeMAPS v02</strong> via OpenSmile<br>Emotion dims: <strong>audeering/wav2vec2-large-robust</strong><br>No semantics used — purely paralinguistic (except for the Hume tab).</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# TTS Expressivity Analysis")
st.markdown("*Purely acoustic analysis. No semantics and no transcripts*")
st.markdown("---")

ready_files = [(model_names[i], uploaded_files[i]) for i in range(num_models) if uploaded_files[i] is not None]

if not analyze:
    # Landing state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="metric-card">
            <h4>What this measures</h4>
            <div style="font-size:0.85rem;color:#a0a0cc;margin-top:0.5rem">
            Pitch range & variability<br>Loudness dynamics<br>Voice quality (shimmer, HNR)<br>Arousal & valence scores
            </div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <h4>Why it's different</h4>
            <div style="font-size:0.85rem;color:#a0a0cc;margin-top:0.5rem">
            Zero semantic contamination.<br>Models are audio-only.<br>Same script, different expressivity — fairly compared.
            </div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <h4>How to use</h4>
            <div style="font-size:0.85rem;color:#a0a0cc;margin-top:0.5rem">
            1. Upload one WAV per TTS model<br>2. Name each model<br>3. Click ▶ Run Analysis
            </div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="info-box">👈 Configure your models in the sidebar, then click <strong>Run Analysis</strong>.</div>', unsafe_allow_html=True)
    st.stop()

if len(ready_files) < 2:
    st.warning("Please upload at least 2 audio files to compare.")
    st.stop()

# ── Run analysis ───────────────────────────────────────────────────────────────
smile = load_smile()
arousal_pipe = load_arousal_model() if use_arousal else None

results = {}  # name → {feats, components, arousal}

progress = st.progress(0, text="Extracting acoustic features…")
for idx, (name, file) in enumerate(ready_files):
    audio_bytes = file.read()
    progress.progress((idx) / len(ready_files), text=f"Processing {name}…")

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

progress.progress(1.0, text="Done!")
progress.empty()

model_list = list(results.keys())
colors = {name: ACCENT_COLORS[i % len(ACCENT_COLORS)] for i, name in enumerate(model_list)}

# ── Composite score ────────────────────────────────────────────────────────────
# Z-score each component across models, then average → composite expressivity index
comp_matrix = {}
for name, r in results.items():
    comp_matrix[name] = r["components"]

df_comp = pd.DataFrame(comp_matrix).T  # models × features

# Normalize: some features like HNR should be inverted? No — higher HNR = cleaner voice,
# but we want expressivity, so we keep directionality as-is except HNR (more monotone = higher HNR)
# Invert HNR: more breathy/dynamic voices are more expressive
df_norm = df_comp.copy()
df_norm["HNR (voice quality)"] = -df_norm["HNR (voice quality)"]  # invert

df_zscored = (df_norm - df_norm.mean()) / (df_norm.std() + 1e-8)
composite = df_zscored.mean(axis=1)

# If arousal available, blend it in
if use_arousal:
    arousal_scores = {name: (results[name]["arousal"] or {}).get("arousal", 0) for name in model_list}
    arousal_series = pd.Series(arousal_scores)
    arousal_z = (arousal_series - arousal_series.mean()) / (arousal_series.std() + 1e-8)
    composite = 0.7 * composite + 0.3 * arousal_z

composite = composite.sort_values(ascending=False)
winner = composite.index[0]


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Rankings", "📊 Acoustic Features", "🧠 Arousal & Emotion", "🦨 Hume Emotion", "🔬 Raw Data"])


# ── TAB 1: Rankings ────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Overall Expressivity Ranking")
    st.markdown("*Composite score: pitch range/variability, loudness dynamics, shimmer" +
                (" + arousal*" if use_arousal else "*"))

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
        st.markdown('<div class="info-box">Score is a z-scored composite — relative within this session. Higher = more expressive.</div>', unsafe_allow_html=True)

    with col_radar:
        # Radar chart of normalized component scores
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
        st.plotly_chart(fig_radar, width='stretch')

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

    # Grouped bar chart
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
    st.plotly_chart(fig_bar, width='stretch')

    st.markdown("### Feature Explanations")
    feat_info = {
        "Pitch Range (F0 range)": ("Semitones between lowest & highest pitch", "Higher = wider melodic range → more expressive"),
        "Pitch Variability (F0 std)": ("Normalized std dev of pitch", "Higher = more dynamic intonation"),
        "Loudness Variability": ("Normalized std dev of loudness (sone)", "Higher = more dramatic volume changes"),
        "Shimmer (local)": ("dB variation between consecutive glottal cycles", "Some shimmer = natural, expressive voice"),
        "HNR (voice quality)": ("Harmonics-to-noise ratio in dB", "Lower HNR here = breathy, more expressive (inverted in composite)"),
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
        st.info("Enable the Arousal Model toggle in the sidebar to see this section.")
    else:
        st.markdown("### Arousal · Valence · Dominance")
        st.markdown("*From `audeering/wav2vec2-large-robust` — audio-only, no text used.*")

        dims = ["arousal", "valence", "dominance"]
        avd_data = {}
        for name in model_list:
            ar = results[name]["arousal"] or {}
            avd_data[name] = {d: ar.get(d, 0) for d in dims}

        df_avd = pd.DataFrame(avd_data).T

        # Scatter: Arousal vs Valence (Russell's circumplex)
        st.markdown("#### Arousal vs Valence (Russell's Circumplex)")
        fig_scatter = go.Figure()
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
        # Quadrant lines
        fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="#2a2a4a")
        fig_scatter.add_vline(x=0.5, line_dash="dash", line_color="#2a2a4a")

        for label, x, y in [("Excited (high energy, positive emotion)", 0.75, 0.75), ("Calm (low energy, positive emotion)", 0.75, 0.25),
                              ("Anxious (high energy, negative emotion)", 0.25, 0.75), ("Sad (low energy, negative emotion)", 0.25, 0.25)]:
            fig_scatter.add_annotation(x=x, y=y, text=label, showarrow=False,
                                       font=dict(color="#3a3a6a", size=11))
        fig_scatter.update_layout(
            **PLOTLY_THEME,
            height=380,
            showlegend=False,
            margin=dict(l=40, r=40, t=20, b=40),
        )
        fig_scatter.update_yaxes(title="Arousal →", range=[0, 1], gridcolor="#1e1e3a")
        st.plotly_chart(fig_scatter, width='stretch')

        # AVD bar chart
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
            **PLOTLY_THEME,
            barmode="group",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(font=dict(color="#a0a0cc")),
        )
        st.plotly_chart(fig_avd, width='stretch')

        st.markdown('<div class="info-box">Arousal ≈ energy/activation level. High arousal = excited, animated speech. This is the strongest single proxy for TTS expressivity.</div>', unsafe_allow_html=True)


with tab4:
    st.markdown("### Raw Hume Emotional Expressivity Analysis")
    st.markdown("Prosody-based emotion analysis from Hume AI — no text used.")

    with st.status("Processing with Hume AI...", expanded=True) as status:
        try:
            st.write("📤 Uploading files to Hume...")

            all_hume_results = {}

            for name, file_obj in ready_files:
                audio_bytes = results[name]["audio"]

                # Write bytes to a temp file so Hume can read it
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                try:
                    configs = InferenceBaseRequest(
                        models=Models(prosody=Prosody(granularity="utterance"))
                    )
                    with open(tmp_path, "rb") as f:
                        job_id = client.expression_measurement.batch.start_inference_job_from_local_file(
                            file=[f],
                            json=configs
                        )
                    st.write(f"✅ Job started for **{name}** (ID: `{job_id}`)")

                    # Poll until done
                    while True:
                        job_status = client.expression_measurement.batch.get_job_details(id=job_id)
                        current_state = job_status.state.status
                        if current_state == "COMPLETED":
                            break
                        elif current_state == "FAILED":
                            st.error(f"Hume job failed for {name}.")
                            break
                        else:
                            st.write(f"⏳ {name}: {current_state}...")
                            time.sleep(3)

                    predictions = client.expression_measurement.batch.get_job_predictions(job_id)
                    all_hume_results[name] = predictions

                finally:
                    os.unlink(tmp_path)

            status.update(label="Hume Analysis Complete!", state="complete")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # ── Display results per model ──────────────────────────────────
    for name, predictions in all_hume_results.items():
        st.divider()
        st.subheader(f"🎙️ {name}")

        all_segment_emotions = []
        for result in predictions:
            group_predictions = result.results.predictions[0].models.prosody.grouped_predictions
            for group in group_predictions:
                for prediction in group.predictions:
                    seg_dict = {e.name: e.score for e in prediction.emotions}
                    all_segment_emotions.append(seg_dict)

        if all_segment_emotions:
            df_all = pd.DataFrame(all_segment_emotions)
            global_means = df_all.mean().sort_values(ascending=False)
            global_sorted = global_means.sort_values(ascending=False)

            st.write("**Top 3 dominant emotions (averaged across utterances):**")
            cols = st.columns(3)
            for i in range(3):
                cols[i].metric(
                    label=global_sorted.index[i],
                    value=f"{global_sorted.values[i]:.2%}"
                )

            # Bar chart of top 10
            fig_hume = go.Figure(go.Bar(
                x=global_means.head(10).index.tolist(),
                y=global_means.head(10).values.tolist(),
                marker_color=colors.get(name, "#cc44ff"),
                opacity=0.85,
            ))
            fig_hume.update_layout(
                **PLOTLY_THEME,
                height=300,
                margin=dict(l=20, r=20, t=20, b=60),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_hume, use_container_width=True)





# ── TAB 5: Raw Data ────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### Raw eGeMAPS Features")
    st.markdown("Full feature vector for each model (203 functionals from eGeMAPS v02).")

    selected = st.selectbox("Select model to inspect", model_list, key="raw_data_model_select")
    feats_df = results[selected]["feats"].to_frame(name="Value")
    feats_df.columns = ["Value"]
    feats_df.index.name = "Feature"
    st.dataframe(feats_df.style.format("{:.5f}"), width='stretch', height=400)

    # Download
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
        st.dataframe(pd.DataFrame(avd_rows).set_index("Model"), width='stretch')