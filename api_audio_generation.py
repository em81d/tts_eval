"""
TTS Studio — Multi-Provider Text-to-Speech Comparison Tool
Providers: ElevenLabs, Hume, Google Gemini, Microsoft Azure, Cartesia Sonic 3,
           Deepgram Aura, Fish Audio, Neuphonic, Inworld, Async AI

Run:  streamlit run tts_studio.py
"""

import os
import io
import wave
import struct
import zipfile
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────────
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")
def get_secret(key):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TTS Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Light-theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Force light theme variables */
  :root {
    --bg: #f8f8f6;
    --surface: #ffffff;
    --border: #e2e2df;
    --accent: #2563eb;
    --accent-light: #dbeafe;
    --text: #1a1a1a;
    --text-muted: #6b7280;
    --success: #16a34a;
    --error: #dc2626;
    --warning: #d97706;
    --radius: 10px;
    --shadow: 0 1px 4px rgba(0,0,0,.08);
  }

  /* Global reset to light */
  .stApp { background: var(--bg) !important; color: var(--text) !important; }
  .stSidebar { background: var(--surface) !important; border-right: 1px solid var(--border); }
  .block-container { padding-top: 1.5rem !important; }

  /* Provider cards */
  .provider-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
    transition: box-shadow .2s;
  }
  .provider-card:hover { box-shadow: 0 3px 12px rgba(0,0,0,.1); }

  .provider-header {
    display: flex;
    align-items: center;
    gap: .6rem;
    margin-bottom: .9rem;
  }
  .provider-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -.01em;
  }
  .provider-badge {
    font-size: .7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 99px;
    background: var(--accent-light);
    color: var(--accent);
  }
  .badge-free { background: #dcfce7; color: #15803d; }
  .badge-paid { background: #fef9c3; color: #854d0e; }

  /* Status messages */
  .status-ok   { color: var(--success); font-size: .82rem; font-weight: 500; }
  .status-err  { color: var(--error);   font-size: .82rem; font-weight: 500; }
  .status-warn { color: var(--warning); font-size: .82rem; font-weight: 500; }

  /* Section heading */
  .section-title {
    font-size: .75rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: .4rem;
  }

  /* Streamlit button overrides */
  .stButton > button {
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: .85rem !important;
  }

  /* Sidebar title */
  .sidebar-title {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -.02em;
    color: var(--text);
    margin-bottom: .25rem;
  }
  .sidebar-sub {
    font-size: .8rem;
    color: var(--text-muted);
    margin-bottom: 1.2rem;
  }

  /* Download all button */
  .download-all-btn > button {
    background: var(--accent) !important;
    color: white !important;
    width: 100% !important;
    padding: .6rem !important;
    font-size: 1rem !important;
  }

  /* Audio player compact */
  audio { width: 100%; margin-top: .4rem; }

  /* Key missing warning */
  .key-missing {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 6px;
    padding: .4rem .75rem;
    font-size: .8rem;
    color: #991b1b;
    margin-top: .3rem;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def mp3_to_wav_fallback(mp3_bytes: bytes) -> bytes:
    """Try pydub conversion; fall back to raw bytes if unavailable."""
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        return buf.getvalue()
    except Exception:
        # Return MP3 bytes as-is; Streamlit audio player can still play it
        return mp3_bytes


def get_env(key: str) -> Optional[str]:
    """Return env var or None."""
    val = os.getenv(key, "").strip()
    return val if val else None


def missing_key_html(key_name: str) -> str:
    return f'<div class="key-missing">⚠️ Missing <code>{key_name}</code> in .env — set it to enable this provider.</div>'


# Session state: store generated audio bytes per provider
if "audio_store" not in st.session_state:
    st.session_state.audio_store = {}


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-title">🎙️ TTS Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Multi-provider speech comparison</div>', unsafe_allow_html=True)

    st.markdown("#### 📝 Prompt")
    prompt_text = st.text_area(
        "Text to synthesize",
        value="Hello! I'm demonstrating text-to-speech synthesis. How does this voice sound to you?",
        height=160,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("API keys are loaded from your `.env` file. Providers with missing keys show an error.")


# ══════════════════════════════════════════════════════════════════════════════
# Page header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## TTS Provider Comparison")
st.caption("Generate speech from all providers simultaneously.")
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# Provider definitions
# (name, env_key, badge, render_fn)
# ══════════════════════════════════════════════════════════════════════════════

def render_elevenlabs(prompt: str):
    """ElevenLabs — audio tags for emotion, voice selector."""
    key = get_secret('ELEVENLABS_API_KEY')

    VOICES = {
        "Rachel (Female)":   "21m00Tcm4TlvDq8ikWAM",
        "Domi (Female)":     "AZnzlk1XvdvUeBnXmlld",
        "Bella (Female)":    "EXAVITQu4vr4xnSDxMaL",
        "Antoni (Male)":     "ErXwobaYiN019PkySvjV",
        "Elli (Female)":     "MF3mGyEYCl7XYWbV9V6O",
        "Josh (Male)":       "TxGEqnHWrfWFTfGW9XjX",
        "Arnold (Male)":     "VR6AewLTigWG4xSOukaG",
        "Adam (Male)":       "pNInz6obpgDQGcFmaJgB",
        "Sam (Male)":        "yoZ06aMxZJJ28mfd3POQ",
    }
    EMOTION_TAGS = [
        "None", "[laughs]", "[whispers]", "[sighs]", "[excited]",
        "[sad]", "[angry]", "[surprised]", "[nervous]", "[cheerful]",
    ]

    c1, c2 = st.columns(2)
    with c1:
        voice_label = st.selectbox("Voice", list(VOICES.keys()), key="el_voice")
    with c2:
        emotion_tag = st.selectbox("Emotion tag (prepended)", EMOTION_TAGS, key="el_emotion")

    if not key:
        st.markdown(missing_key_html("ELEVENLABS_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key="el_gen")
    with col_dl:
        dl_placeholder = st.empty()

    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True)
            return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True)
            return
        try:
            import requests
            tag = "" if emotion_tag == "None" else emotion_tag + " "
            text = tag + prompt
            voice_id = VOICES[voice_label]
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {"xi-api-key": key, "Content-Type": "application/json",
                       "Accept": "audio/mpeg"}
            payload = {
                "text": text,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            }
            with st.spinner("Generating…"):
                resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ API error {resp.status_code}: {resp.text[:200]}</span>',
                                 unsafe_allow_html=True)
                return
            wav = mp3_to_wav_fallback(resp.content)
            st.session_state.audio_store["ElevenLabs"] = wav
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True)

    # Show audio + download if available
    if "ElevenLabs" in st.session_state.audio_store:
        wav = st.session_state.audio_store["ElevenLabs"]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, "elevenlabs.wav", "audio/wav", key="el_dl")


def render_hume(prompt: str):
    """Hume Octave TTS — acting instructions (emotion/style description)."""
    key = get_secret('HUME_API_KEY')

    ACTING_PRESETS = [
        "None",
        "happy, upbeat",
        "melancholic, slow",
        "excited, energetic",
        "calm, reassuring",
        "whispering, secretive",
        "angry, frustrated",
        "nervous, hesitant",
        "confident, professional",
        "warm, friendly",
        "sarcastic, dry",
    ]
    HUME_LIBRARY_VOICES = [
        "Auto-generate (no voice)",  # omit voice field entirely
        "ITO", "KORA", "DACHER", "AURA", "FINN", "WHIMSY", "STELLA", "SUNNY",
    ]

    c1, c2 = st.columns(2)
    with c1:
        voice_label = st.selectbox("Voice (Hume Library)", HUME_LIBRARY_VOICES, key="hume_voice")
    with c2:
        acting = st.selectbox("Acting instruction", ACTING_PRESETS, key="hume_acting")

    custom_voice_id = st.text_input(
        "Or paste custom voice ID from your Hume account (overrides dropdown)",
        key="hume_custom_voice",
        placeholder="e.g. abc123… (leave blank to use dropdown)"
    )
    custom_acting = st.text_input("Or type custom acting instruction", key="hume_custom",
                                   placeholder="e.g. 'frightened, rushed'")

    if not key:
        st.markdown(missing_key_html("HUME_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key="hume_gen")
    with col_dl:
        dl_placeholder = st.empty()

    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True)
            return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True)
            return
        try:
            import requests, base64
            effective_acting = custom_acting.strip() or (None if acting == "None" else acting)

            headers = {"X-Hume-Api-Key": key, "Content-Type": "application/json"}
            utterance: dict = {"text": prompt}

            custom_vid = custom_voice_id.strip()
            if custom_vid:
                utterance["voice"] = {"id": custom_vid}
            elif voice_label != "Auto-generate (no voice)":
                utterance["voice"] = {
                    "name": voice_label,
                    "provider": "HUME_AI",   # ← this was the missing piece
                }
            # else: omit voice → Hume auto-generates

            if effective_acting:
                utterance["description"] = effective_acting

            body = {"utterances": [utterance]}

            with st.spinner("Generating…"):
                resp = requests.post(
                    "https://api.hume.ai/v0/tts",
                    json=body, headers=headers, timeout=30
                )
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ API error {resp.status_code}: {resp.text[:300]}</span>',
                                 unsafe_allow_html=True)
                return
            data = resp.json()
            # Hume returns base64-encoded audio
            audio_b64 = data["generations"][0]["audio"]
            audio_bytes = base64.b64decode(audio_b64)
            wav = mp3_to_wav_fallback(audio_bytes)
            st.session_state.audio_store["Hume"] = wav
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True)
            traceback.print_exc()

    if "Hume" in st.session_state.audio_store:
        wav = st.session_state.audio_store["Hume"]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, "hume.wav", "audio/wav", key="hume_dl")


def render_google_gemini(prompt: str):
    """Google Gemini TTS — style prompt + voice selection."""
    key = get_secret('GOOGLE_API_KEY')

    VOICES = [
        "Aoede", "Charon", "Fenrir", "Kore", "Leda", "Orus",
        "Puck", "Schedar", "Sulafat", "Umbriel", "Zephyr",
        "Achird", "Algieba", "Alnilam", "Autonoe", "Callirrhoe",
        "Despina", "Enceladus", "Erinome", "Gacrux", "Iocaste",
        "Laomedeia", "Pulcherrima", "Rasalgethi", "Sadachbia",
        "Sadaltager", "Vindemiatrix", "Zubenelgenubi",
    ]
    STYLE_PRESETS = [
        "None (no style prompt)",
        "Say in an excited, upbeat tone",
        "Say calmly and slowly",
        "Say in a spooky whisper",
        "Say with sadness and empathy",
        "Say enthusiastically like a sports announcer",
        "Say in a professional newscast style",
        "Say warmly, like speaking to a friend",
        "Say dramatically, like a movie trailer",
    ]

    c1, c2 = st.columns(2)
    with c1:
        voice = st.selectbox("Voice", VOICES, key="gg_voice")
    with c2:
        style_preset = st.selectbox("Style prompt", STYLE_PRESETS, key="gg_style")
    custom_style = st.text_input("Or type custom style prompt", key="gg_custom",
                                  placeholder='e.g. "Say thoughtfully, with long pauses"')

    if not key:
        st.markdown(missing_key_html("GOOGLE_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key="gg_gen")
    with col_dl:
        dl_placeholder = st.empty()

    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True)
            return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True)
            return
        try:
            import requests, base64, struct

            eff_style = custom_style.strip()
            if not eff_style and style_preset != "None (no style prompt)":
                eff_style = style_preset
            full_text = f"{eff_style}: {prompt}" if eff_style else prompt

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={key}"
            body = {
                "contents": [{"parts": [{"text": full_text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
                    }
                }
            }
            with st.spinner("Generating…"):
                resp = requests.post(url, json=body, timeout=60)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ API error {resp.status_code}: {resp.text[:300]}</span>',
                                 unsafe_allow_html=True)
                return
            data = resp.json()
            audio_b64 = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
            pcm_bytes = base64.b64decode(audio_b64)
            wav = pcm_to_wav(pcm_bytes, sample_rate=24000, channels=1, sample_width=2)
            st.session_state.audio_store["Google Gemini"] = wav
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True)
            traceback.print_exc()

    if "Google Gemini" in st.session_state.audio_store:
        wav = st.session_state.audio_store["Google Gemini"]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, "google_gemini.wav", "audio/wav", key="gg_dl")


def render_azure(prompt: str):
    """Microsoft Azure Speech — SSML emotion styles."""
    key = get_env("AZURE_SPEECH_KEY")
    region = get_env("AZURE_SPEECH_REGION") or "westus"

    VOICES = {
        "Jenny (Female, en-US)":    ("en-US-JennyNeural",    "en-US"),
        "Guy (Male, en-US)":        ("en-US-GuyNeural",      "en-US"),
        "Aria (Female, en-US)":     ("en-US-AriaNeural",     "en-US"),
        "Davis (Male, en-US)":      ("en-US-DavisNeural",    "en-US"),
        "Jane (Female, en-US)":     ("en-US-JaneNeural",     "en-US"),
        "Jason (Male, en-US)":      ("en-US-JasonNeural",    "en-US"),
        "Sara (Female, en-US)":     ("en-US-SaraNeural",     "en-US"),
        "Tony (Male, en-US)":       ("en-US-TonyNeural",     "en-US"),
        "Ryan (Male, en-GB)":       ("en-GB-RyanNeural",     "en-GB"),
        "Sonia (Female, en-GB)":    ("en-GB-SoniaNeural",    "en-GB"),
    }
    # Style support varies by voice; these are commonly available
    STYLES = [
        "general",
        "cheerful",
        "sad",
        "angry",
        "fearful",
        "disgruntled",
        "serious",
        "affectionate",
        "gentle",
        "embarrassed",
        "empathetic",
        "newscast",
        "customerservice",
        "whispering",
        "shouting",
        "lyrical",
    ]

    c1, c2 = st.columns(2)
    with c1:
        voice_label = st.selectbox("Voice", list(VOICES.keys()), key="az_voice")
    with c2:
        style = st.selectbox("Speaking style", STYLES, key="az_style")

    style_degree = st.slider("Style intensity", 0.1, 2.0, 1.0, 0.1, key="az_degree")

    if not key:
        st.markdown(missing_key_html("AZURE_SPEECH_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key="az_gen")
    with col_dl:
        dl_placeholder = st.empty()

    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True)
            return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True)
            return
        try:
            import requests
            voice_name, lang = VOICES[voice_label]

            # Build SSML
            ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis'
  xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='{lang}'>
  <voice name='{voice_name}'>
    <mstts:express-as style='{style}' styledegree='{style_degree:.1f}'>
      {prompt}
    </mstts:express-as>
  </voice>
</speak>"""

            token_url = f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
            with st.spinner("Getting token…"):
                tok_resp = requests.post(token_url,
                                         headers={"Ocp-Apim-Subscription-Key": key}, timeout=10)
            if tok_resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ Token error {tok_resp.status_code}</span>',
                                 unsafe_allow_html=True)
                return
            token = tok_resp.text

            tts_url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
                "User-Agent": "TTS-Studio",
            }
            with st.spinner("Synthesizing…"):
                resp = requests.post(tts_url, data=ssml.encode("utf-8"), headers=headers, timeout=30)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ TTS error {resp.status_code}: {resp.text[:200]}</span>',
                                 unsafe_allow_html=True)
                return
            # Azure returns WAV directly with riff format
            st.session_state.audio_store["Azure"] = resp.content
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True)
            traceback.print_exc()

    if "Azure" in st.session_state.audio_store:
        wav = st.session_state.audio_store["Azure"]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, "azure.wav", "audio/wav", key="az_dl")


def render_cartesia(prompt: str):
    """Cartesia Sonic 3 — emotion & speed controls."""
    key = get_secret('CARTESIA_API_KEY')

    # Curated default voices from Cartesia library
    VOICES = {
        "Barbershop Man":   "a0e99841-438c-4a64-b679-ae501e7d6091",
        "Female Nurse":     "5c42302c-194b-4d0c-ba1a-8cb485c84ab9",
        "Sweet Lady":       "e3827ec5-697a-4b7c-9704-1a23041bbc51",
        "Friendly Reading": "79f8b5fb-2cc8-479a-80df-29f7a7cf1a3e",
        "Deep Male":        "ee7ea9f8-c0c1-498c-9279-764d6b56d189",
        "Calm Lady":        "b7d50908-b17c-442d-ad8d-810c63997ed9",
        "British Reporter": "71a7ad14-091d-4441-9c30-be5fc3e25d32",
    }
    EMOTIONS = [
        "neutral", "angry", "excited", "content", "sad", "scared",
        "happy", "enthusiastic", "elated", "euphoric", "triumphant",
        "surprised", "curious", "calm", "grateful", "sympathetic",
        "sarcastic", "dejected", "melancholic",
    ]
    SPEEDS = {"Slowest": 0.6, "Slow": 0.8, "Normal": 1.0, "Fast": 1.2, "Fastest": 1.5}

    c1, c2, c3 = st.columns(3)
    with c1:
        voice_label = st.selectbox("Voice", list(VOICES.keys()), key="ca_voice")
    with c2:
        emotion = st.selectbox("Emotion", EMOTIONS, key="ca_emotion")
    with c3:
        speed_label = st.selectbox("Speed", list(SPEEDS.keys()), index=2, key="ca_speed")

    if not key:
        st.markdown(missing_key_html("CARTESIA_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key="ca_gen")
    with col_dl:
        dl_placeholder = st.empty()

    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True)
            return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True)
            return
        try:
            import requests
            voice_id = VOICES[voice_label]
            speed_val = SPEEDS[speed_label]

            url = "https://api.cartesia.ai/tts/bytes"
            headers = {
                "X-API-Key": key,
                "Cartesia-Version": "2025-04-16",
                "Content-Type": "application/json",
            }
            body = {
                "model_id": "sonic-3",
                "transcript": prompt,
                "voice": {"mode": "id", "id": voice_id},
                "output_format": {"container": "wav", "encoding": "pcm_f32le", "sample_rate": 44100},
                "generation_config": {
                    "speed": speed_val,
                    "emotion": emotion,
                },
            }
            with st.spinner("Generating…"):
                resp = requests.post(url, json=body, headers=headers, timeout=30)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ API error {resp.status_code}: {resp.text[:300]}</span>',
                                 unsafe_allow_html=True)
                return
            st.session_state.audio_store["Cartesia"] = resp.content
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True)
            traceback.print_exc()

    if "Cartesia" in st.session_state.audio_store:
        wav = st.session_state.audio_store["Cartesia"]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, "cartesia.wav", "audio/wav", key="ca_dl")


def render_deepgram(prompt: str):
    """Deepgram Aura — voice selector (no explicit emotion params in REST API)."""
    key = get_secret('DEEPGRAM_API_KEY')

    VOICES = [
        "aura-2-thalia-en", "aura-2-andromeda-en", "aura-2-helena-en",
        "aura-2-apollo-en", "aura-2-orion-en",  "aura-2-zeus-en",
        "aura-2-luna-en",   "aura-2-stella-en", "aura-2-asteria-en",
        "aura-asteria-en",  "aura-luna-en",     "aura-stella-en",
        "aura-athena-en",   "aura-hera-en",     "aura-orion-en",
        "aura-arcas-en",    "aura-perseus-en",  "aura-angus-en",
        "aura-orpheus-en",  "aura-helios-en",   "aura-zeus-en",
    ]
    ENCODING_OPTIONS = {"WAV/PCM (linear16)": "linear16", "MP3": "mp3", "Opus": "opus"}

    c1, c2 = st.columns(2)
    with c1:
        voice = st.selectbox("Voice model", VOICES, key="dg_voice")
    with c2:
        enc_label = st.selectbox("Format", list(ENCODING_OPTIONS.keys()), key="dg_enc")

    if not key:
        st.markdown(missing_key_html("DEEPGRAM_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key="dg_gen")
    with col_dl:
        dl_placeholder = st.empty()

    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True)
            return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True)
            return
        try:
            import requests
            encoding = ENCODING_OPTIONS[enc_label]
            params = {"model": voice, "encoding": encoding}
            if encoding == "linear16":
                params["sample_rate"] = "24000"
                params["container"] = "wav"
            headers = {
                "Authorization": f"Token {key}",
                "Content-Type": "application/json",
            }
            with st.spinner("Generating…"):
                resp = requests.post(
                    "https://api.deepgram.com/v1/speak",
                    params=params,
                    json={"text": prompt},
                    headers=headers,
                    timeout=30,
                )
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ API error {resp.status_code}: {resp.text[:300]}</span>',
                                 unsafe_allow_html=True)
                return
            audio_bytes = resp.content
            if encoding == "mp3":
                audio_bytes = mp3_to_wav_fallback(audio_bytes)
            st.session_state.audio_store["Deepgram"] = audio_bytes
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True)
            traceback.print_exc()

    if "Deepgram" in st.session_state.audio_store:
        wav = st.session_state.audio_store["Deepgram"]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, "deepgram.wav", "audio/wav", key="dg_dl")


def render_fish_audio(prompt: str):
    """Fish Audio — emotion tags (inline [tag] syntax), voice selector."""
    key = get_secret('FISH_AUDIO_API_KEY')

    # Curated well-known public voices on Fish Audio
    VOICES = {
        "Default (no reference)": None,
        "Custom reference ID…":   "__custom__",
    }
    EMOTION_TAGS = [
        "None",
        "[laughing]", "[chuckling]", "[sobbing]", "[crying loudly]",
        "[sighing]", "[panting]", "[groaning]", "[whispers]",
        "[excited]", "[angry]", "[sad]", "[happy]",
        "[super happy]", "[long pause]", "[pause]",
    ]

    c1, c2 = st.columns(2)
    with c1:
        voice_label = st.selectbox("Voice preset", list(VOICES.keys()), key="fa_voice")
    with c2:
        emotion_tag = st.selectbox("Inline emotion tag", EMOTION_TAGS, key="fa_emotion")

    custom_ref = ""
    if voice_label == "Custom reference ID…":
        custom_ref = st.text_input("Voice reference ID (from fish.audio)", key="fa_ref_id",
                                    placeholder="e.g. 802e3bc2b27e49c2995d23ef70e6ac89")

    if not key:
        st.markdown(missing_key_html("FISH_AUDIO_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key="fa_gen")
    with col_dl:
        dl_placeholder = st.empty()

    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True)
            return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True)
            return
        try:
            import requests, json as _json
            tag = "" if emotion_tag == "None" else emotion_tag + " "
            text = tag + prompt

            ref_id = None
            if voice_label == "Custom reference ID…" and custom_ref.strip():
                ref_id = custom_ref.strip()

            payload = {
                "text": text,
                "format": "wav",
                "latency": "balanced",
            }
            if ref_id:
                payload["reference_id"] = ref_id

            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            with st.spinner("Generating…"):
                resp = requests.post(
                    "https://api.fish.audio/v1/tts",
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ API error {resp.status_code}: {resp.text[:300]}</span>',
                                 unsafe_allow_html=True)
                return
            st.session_state.audio_store["Fish Audio"] = resp.content
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True)
            traceback.print_exc()

    if "Fish Audio" in st.session_state.audio_store:
        wav = st.session_state.audio_store["Fish Audio"]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, "fish_audio.wav", "audio/wav", key="fa_dl")


def render_neuphonic(prompt: str):
    """Neuphonic — speed, pitch, language controls."""
    key = get_secret('NEUPHONIC_API_KEY')

    LANGUAGES = {
        "English (US)": "en",
        "English (UK)": "en-gb",
        "Spanish":      "es",
        "French":       "fr",
        "German":       "de",
        "Portuguese":   "pt",
        "Italian":      "it",
        "Dutch":        "nl",
        "Polish":       "pl",
        "Arabic":       "ar",
    }

    c1, c2 = st.columns(2)
    with c1:
        lang_label = st.selectbox("Language", list(LANGUAGES.keys()), key="neu_lang")
    with c2:
        voice_id_input = st.text_input(
            "Voice ID (from Neuphonic dashboard, leave blank for default)",
            key="neu_voice_id",
            placeholder="e.g. 8e9c4bc8-3979-48ab-8626-df53befc2090"
        )

    speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.05, key="neu_speed")

    if not key:
        st.markdown(missing_key_html("NEUPHONIC_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key="neu_gen")
    with col_dl:
        dl_placeholder = st.empty()

    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True)
            return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True)
            return
        try:
            from pyneuphonic import Neuphonic, TTSConfig
        except ImportError:
            status.markdown(
                '<span class="status-err">❌ pyneuphonic not installed. Run: <code>pip install pyneuphonic</code></span>',
                unsafe_allow_html=True
            )
            return
        try:
            lang_code = LANGUAGES[lang_label]
            vid = voice_id_input.strip() or None

            client = Neuphonic(api_key=key)
            sse = client.tts.SSEClient()

            tts_config = TTSConfig(
                speed=speed,
                lang_code=lang_code,
                voice_id=vid,
                sampling_rate=22050,
            )

            with st.spinner("Generating…"):
                response = sse.send(prompt, tts_config=tts_config)

            pcm_chunks = []
            for chunk in response:
                audio_chunk = getattr(getattr(chunk, "data", None), "audio", None)
                if audio_chunk is not None:
                    pcm_chunks.append(bytes(audio_chunk))

            if not pcm_chunks:
                status.markdown('<span class="status-err">❌ No audio returned from Neuphonic</span>',
                                 unsafe_allow_html=True)
                return

            pcm_bytes = b"".join(pcm_chunks)
            wav = pcm_to_wav(pcm_bytes, sample_rate=22050, channels=1, sample_width=2)
            st.session_state.audio_store["Neuphonic"] = wav
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True)
            traceback.print_exc()

    if "Neuphonic" in st.session_state.audio_store:
        wav = st.session_state.audio_store["Neuphonic"]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, "neuphonic.wav", "audio/wav", key="neu_dl")


def render_inworld(prompt: str):
    """Inworld AI TTS — character/voice selection."""
    key = get_secret('INWORLD_API_KEY')

    st.info("Inworld TTS uses a character-based voice model. Provide your Inworld workspace ID and character name in the .env file, or enter them below.")

    c1, c2 = st.columns(2)
    with c1:
        workspace_id = st.text_input("Workspace ID",
                                      value=get_env("INWORLD_WORKSPACE_ID") or "",
                                      key="iw_workspace",
                                      placeholder="workspaces/my-workspace")
    with c2:
        character_name = st.text_input("Character name",
                                        value=get_env("INWORLD_CHARACTER") or "",
                                        key="iw_character",
                                        placeholder="characters/my-character")

    if not key:
        st.markdown(missing_key_html("INWORLD_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key="iw_gen")
    with col_dl:
        dl_placeholder = st.empty()

    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True)
            return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True)
            return
        try:
            import requests, base64
            headers = {
                "Authorization": f"Basic {key}",
                "Content-Type": "application/json",
            }
            body = {
                "text": prompt,
                "character": character_name or "characters/default",
            }
            with st.spinner("Generating…"):
                resp = requests.post(
                    "https://studio.inworld.ai/v1/tts:synthesize",
                    json=body, headers=headers, timeout=30
                )
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ API error {resp.status_code}: {resp.text[:300]}</span>',
                                 unsafe_allow_html=True)
                return
            data = resp.json()
            audio_b64 = data.get("audioContent", "")
            audio_bytes = base64.b64decode(audio_b64)
            wav = mp3_to_wav_fallback(audio_bytes)
            st.session_state.audio_store["Inworld"] = wav
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True)
            traceback.print_exc()

    if "Inworld" in st.session_state.audio_store:
        wav = st.session_state.audio_store["Inworld"]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, "inworld.wav", "audio/wav", key="iw_dl")


def render_async_ai(prompt: str):
    """Async AI TTS — voice and style controls."""
    key = get_secret('ASYNC_API_KEY')

    MODELS = {
        "Async Flash v1 (fast)": "async_flash_v1.0",
        "Async v1 (quality)":    "async_v1.0",
    }
    OUTPUT_FORMATS = {
        "WAV / PCM float32 (44.1 kHz)": {
            "container": "wav", "encoding": "pcm_f32le", "sample_rate": 44100
        },
        "WAV / PCM int16 (24 kHz)": {
            "container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000
        },
        "MP3": {
            "container": "mp3", "encoding": "mp3", "sample_rate": 44100
        },
    }

    st.caption(
        "Async uses voice UUIDs. Find yours at [console.async.com](https://console.async.com) "
        "or via the List Voices API. Paste the UUID below."
    )

    c1, c2 = st.columns(2)
    with c1:
        model_label = st.selectbox("Model", list(MODELS.keys()), key="async_model")
    with c2:
        fmt_label = st.selectbox("Output format", list(OUTPUT_FORMATS.keys()), key="async_fmt")

    voice_id = st.text_input(
        "Voice UUID (from Async console)",
        key="async_voice_id",
        placeholder="e.g. e0f39dc4-f691-4e78-bba5-5c636692cc04"
    )

    if not key:
        st.markdown(missing_key_html("ASYNC_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key="async_gen")
    with col_dl:
        dl_placeholder = st.empty()

    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True)
            return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True)
            return
        try:
            import requests
            if not voice_id.strip():
                status.markdown('<span class="status-warn">⚠️ Please enter a voice UUID from your Async console</span>',
                                 unsafe_allow_html=True)
                return
            headers = {
                "x-api-key": key,          # ← was "Authorization: Bearer"
                "version": "v1",
                "Content-Type": "application/json",
            }
            body = {
                "model_id": MODELS[model_label],   # ← was "model": "tts-1"
                "transcript": prompt,               # ← was "input"
                "voice": {
                    "mode": "id",
                    "id": voice_id.strip(),         # ← was a plain string name
                },
                "output_format": OUTPUT_FORMATS[fmt_label],
            }
            with st.spinner("Generating…"):
                resp = requests.post(
                    "https://api.async.com/text_to_speech",   # ← was /v1/audio/speech
                    json=body, headers=headers, timeout=30
                )

            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ API error {resp.status_code}: {resp.text[:300]}</span>',
                                 unsafe_allow_html=True)
                return
            st.session_state.audio_store["Async AI"] = resp.content
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True)
            traceback.print_exc()

    if "Async AI" in st.session_state.audio_store:
        wav = st.session_state.audio_store["Async AI"]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, "async_ai.wav", "audio/wav", key="async_dl")


# ══════════════════════════════════════════════════════════════════════════════
# Provider registry
# ══════════════════════════════════════════════════════════════════════════════

PROVIDERS = [
    {
        "name":   "ElevenLabs",
        "badge":  "Free + Paid",
        "badge_class": "badge-free",
        "emoji":  "🎧",
        "desc":   "Industry-leading expressive TTS. Supports audio emotion tags like [laughs], [whispers]. Uses Turbo v2.5 model.",
        "render": render_elevenlabs,
    },
    {
        "name":   "Hume (Octave)",
        "badge":  "Free + Paid",
        "badge_class": "badge-free",
        "emoji":  "🧠",
        "desc":   "LLM-based TTS that understands emotional context. Supports acting instructions like 'frightened, rushed'.",
        "render": render_hume,
    },
    {
        "name":   "Google Gemini TTS",
        "badge":  "Free tier",
        "badge_class": "badge-free",
        "emoji":  "✨",
        "desc":   "Gemini 2.5 Flash TTS. Natural-language style prompts control tone, accent, pacing and emotion.",
        "render": render_google_gemini,
    },
    {
        "name":   "Microsoft Azure Speech",
        "badge":  "Free tier",
        "badge_class": "badge-free",
        "emoji":  "☁️",
        "desc":   "Neural TTS with SSML emotion styles (cheerful, sad, whispering, etc.) and style degree control.",
        "render": render_azure,
    },
    {
        "name":   "Cartesia Sonic 3",
        "badge":  "Free tier",
        "badge_class": "badge-free",
        "emoji":  "⚡",
        "desc":   "Ultra-low latency (~40ms TTFA). Rich emotion palette (60+ emotions) and speed/volume controls.",
        "render": render_cartesia,
    },
    {
        "name":   "Deepgram Aura",
        "badge":  "$200 free credit",
        "badge_class": "badge-paid",
        "emoji":  "🔊",
        "desc":   "Enterprise-grade TTS (Aura-2). Built for real-time voice agents. 40+ voices with domain-accurate pronunciation.",
        "render": render_deepgram,
    },
    {
        "name":   "Fish Audio",
        "badge":  "Free tier",
        "badge_class": "badge-free",
        "emoji":  "🐟",
        "desc":   "Multilingual TTS with inline emotion tags ([laughing], [whispers], [excited]). Excellent for Asian languages.",
        "render": render_fish_audio,
    },
    {
        "name":   "Neuphonic",
        "badge":  "Free beta",
        "badge_class": "badge-free",
        "emoji":  "🌊",
        "desc":   "Free beta TTS with speed and pitch controls. Multiple languages supported.",
        "render": render_neuphonic,
    },
    {
        "name":   "Inworld AI",
        "badge":  "Free credits",
        "badge_class": "badge-free",
        "emoji":  "🎮",
        "desc":   "Character-based TTS for games and interactive experiences. Requires workspace + character setup.",
        "render": render_inworld,
    },
    {
        "name":   "Async AI",
        "badge":  "Free tier",
        "badge_class": "badge-free",
        "emoji":  "🔄",
        "desc":   "OpenAI-compatible TTS API with speed control. Drop-in replacement for OpenAI TTS.",
        "render": render_async_ai,
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Render all provider cards
# ══════════════════════════════════════════════════════════════════════════════

for provider in PROVIDERS:
    with st.container():
        st.markdown(
            f"""<div class="provider-card">
              <div class="provider-header">
                <span style="font-size:1.3rem">{provider['emoji']}</span>
                <span class="provider-name">{provider['name']}</span>
                <span class="provider-badge {provider['badge_class']}">{provider['badge']}</span>
              </div>
              <div style="font-size:.82rem; color:#6b7280; margin-bottom:.8rem;">{provider['desc']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        # Render controls OUTSIDE the HTML block (Streamlit widgets can't go inside HTML)
        try:
            provider["render"](prompt_text)
        except Exception as e:
            st.error(f"⚠️ Unexpected error in {provider['name']}: {e}")
        st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# Download All button
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## ⬇ Download All Generated Samples")

available = {k: v for k, v in st.session_state.audio_store.items() if v}

if not available:
    st.info("Generate at least one sample above to enable the bulk download.")
else:
    st.success(f"{len(available)} sample(s) ready: {', '.join(available.keys())}")

    def build_zip(audio_dict: dict) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, wav_bytes in audio_dict.items():
                safe_name = name.lower().replace(" ", "_").replace("/", "_")
                zf.writestr(f"{safe_name}.wav", wav_bytes)
        return buf.getvalue()

    zip_bytes = build_zip(available)
    st.download_button(
        label=f"⬇ Download all {len(available)} samples as .zip",
        data=zip_bytes,
        file_name="tts_samples.zip",
        mime="application/zip",
        use_container_width=True,
        key="dl_all",
    )
