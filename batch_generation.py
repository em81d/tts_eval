"""
TTS Studio — Multi-Provider Text-to-Speech Comparison Tool
Providers: ElevenLabs, Hume, Google Gemini, Microsoft Azure, Cartesia Sonic 3,
           Deepgram Aura, Fish Audio, Neuphonic, Inworld, Async AI,
           OpenAI TTS, Murf AI, LMNT, Rime, MiniMax, Smallest AI

Each provider supports multiple independent generation instances — add as many
voice/settings combinations as you want and download them all at once.

Run:  streamlit run tts_studio.py
"""

import os
import io
import wave
import zipfile
import traceback
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────────
load_dotenv()

def get_secret(key):
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        pass
    return os.getenv(key)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TTS Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  :root {
    --bg: #f8f8f6; --surface: #ffffff; --border: #e2e2df;
    --accent: #2563eb; --accent-light: #dbeafe;
    --text: #1a1a1a; --text-muted: #6b7280;
    --success: #16a34a; --error: #dc2626; --warning: #d97706;
    --radius: 10px; --shadow: 0 1px 4px rgba(0,0,0,.08);
  }
  .stApp { background: var(--bg) !important; color: var(--text) !important; }
  .stSidebar { background: var(--surface) !important; border-right: 1px solid var(--border); }
  .block-container { padding-top: 1.5rem !important; }

  .provider-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.25rem 1.5rem;
    margin-bottom: .5rem; box-shadow: var(--shadow); transition: box-shadow .2s;
  }
  .provider-card:hover { box-shadow: 0 3px 12px rgba(0,0,0,.1); }
  .provider-header { display: flex; align-items: center; gap: .6rem; margin-bottom: .5rem; }
  .provider-name { font-size: 1.05rem; font-weight: 700; color: var(--text); letter-spacing: -.01em; }
  .provider-badge {
    font-size: .7rem; font-weight: 600; padding: 2px 8px;
    border-radius: 99px; background: var(--accent-light); color: var(--accent);
  }
  .badge-free { background: #dcfce7; color: #15803d; }
  .badge-paid { background: #fef9c3; color: #854d0e; }

  .instance-card {
    background: #f9fafb; border: 1px solid #e5e7eb;
    border-radius: 8px; padding: .9rem 1rem; margin: .5rem 0;
  }
  .instance-label {
    font-size: .72rem; font-weight: 700; letter-spacing: .07em;
    text-transform: uppercase; color: var(--text-muted); margin-bottom: .5rem;
  }

  .status-ok   { color: var(--success); font-size: .82rem; font-weight: 500; }
  .status-err  { color: var(--error);   font-size: .82rem; font-weight: 500; }
  .status-warn { color: var(--warning); font-size: .82rem; font-weight: 500; }

  .stButton > button { border-radius: 6px !important; font-weight: 600 !important; font-size: .85rem !important; }

  .sidebar-title { font-size: 1.4rem; font-weight: 800; letter-spacing: -.02em; color: var(--text); margin-bottom: .25rem; }
  .sidebar-sub   { font-size: .8rem; color: var(--text-muted); margin-bottom: 1.2rem; }

  audio { width: 100%; margin-top: .4rem; }
  .key-missing {
    background: #fef2f2; border: 1px solid #fecaca; border-radius: 6px;
    padding: .4rem .75rem; font-size: .8rem; color: #991b1b; margin-top: .3rem;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def mp3_to_wav_fallback(mp3_bytes: bytes) -> bytes:
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        return buf.getvalue()
    except Exception:
        return mp3_bytes


def missing_key_html(key_name: str) -> str:
    return f'<div class="key-missing">⚠️ Missing <code>{key_name}</code> in .env — set it to enable this provider.</div>'


# ── Session state ──────────────────────────────────────────────────────────────
if "audio_store" not in st.session_state:
    st.session_state.audio_store = {}           # key: "ProviderName_iid" -> bytes
if "provider_instance_ids" not in st.session_state:
    st.session_state.provider_instance_ids = {} # provider_name -> [0, 1, 2, …]
if "provider_next_iid" not in st.session_state:
    st.session_state.provider_next_iid = {}     # provider_name -> next int to assign


def _init_provider(name: str):
    if name not in st.session_state.provider_instance_ids:
        st.session_state.provider_instance_ids[name] = [0]
        st.session_state.provider_next_iid[name] = 1


def _add_instance(name: str):
    nid = st.session_state.provider_next_iid[name]
    st.session_state.provider_instance_ids[name].append(nid)
    st.session_state.provider_next_iid[name] += 1


def _remove_instance(name: str, iid: int):
    st.session_state.provider_instance_ids[name].remove(iid)
    st.session_state.audio_store.pop(f"{name}_{iid}", None)


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
    st.markdown("#### 🏷️ Label")
    phase_emotion = st.text_input(
        "Phase / emotion tag",
        value="",
        placeholder="e.g. phase1, angry, take2",
        help="Appended to every downloaded filename so you can tell iterations apart.",
        label_visibility="collapsed",
    )
    st.caption("Added to filenames: `elevenlabs_phase1_voice1.wav`")
    st.markdown("---")
    st.caption("API keys are loaded from your `.env` file. Use **＋ Add Voice** under each provider to generate multiple voices in one run.")


# ══════════════════════════════════════════════════════════════════════════════
# Page header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## TTS Provider Comparison")
st.caption("Generate speech from multiple voices per provider. Click **＋ Add Voice** to add more instances.")
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# Provider render functions — each accepts (prompt, iid)
# All widget keys are suffixed with _{iid} for uniqueness.
# Audio is stored as audio_store[f"{ProviderName}_{iid}"]
# ══════════════════════════════════════════════════════════════════════════════

def render_elevenlabs(prompt: str, iid: int = 0, phase: str = ""):
    key = get_secret('ELEVENLABS_API_KEY')
    store_key = f"ElevenLabs_{iid}"
    _phase = f"_{phase}" if phase.strip() else ""

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
    EMOTION_TAGS = ["None", "[laughs]", "[whispers]", "[sighs]", "[excited]",
                    "[sad]", "[angry]", "[surprised]", "[nervous]", "[cheerful]"]

    c1, c2 = st.columns(2)
    with c1:
        voice_label = st.selectbox("Voice", list(VOICES.keys()), key=f"el_voice_{iid}")
    with c2:
        emotion_tag = st.selectbox("Emotion tag", EMOTION_TAGS, key=f"el_emotion_{iid}")

    if not key:
        st.markdown(missing_key_html("ELEVENLABS_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key=f"el_gen_{iid}")
    with col_dl:
        dl_placeholder = st.empty()
    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True); return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True); return
        try:
            import requests
            tag = "" if emotion_tag == "None" else emotion_tag + " "
            voice_id = VOICES[voice_label]
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"}
            payload = {"text": tag + prompt, "model_id": "eleven_turbo_v2_5",
                       "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
            with st.spinner("Generating…"):
                resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ {resp.status_code}: {resp.text[:200]}</span>', unsafe_allow_html=True); return
            st.session_state.audio_store[store_key] = mp3_to_wav_fallback(resp.content)
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True)

    if store_key in st.session_state.audio_store:
        wav = st.session_state.audio_store[store_key]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, f"elevenlabs{_phase}_{iid}.wav", "audio/wav", key=f"el_dl_{iid}")


def render_hume(prompt: str, iid: int = 0, phase: str = ""):
    key = get_secret('HUME_API_KEY')
    store_key = f"Hume_{iid}"
    _phase = f"_{phase}" if phase.strip() else ""

    ACTING_PRESETS = ["None", "happy, upbeat", "melancholic, slow", "excited, energetic",
                      "calm, reassuring", "whispering, secretive", "angry, frustrated",
                      "nervous, hesitant", "confident, professional", "warm, friendly", "sarcastic, dry"]
    HUME_LIBRARY_VOICES = ["Auto-generate (no voice)", "ITO", "KORA", "DACHER", "AURA",
                           "FINN", "WHIMSY", "STELLA", "SUNNY"]

    c1, c2 = st.columns(2)
    with c1:
        voice_label = st.selectbox("Voice (Hume Library)", HUME_LIBRARY_VOICES, key=f"hume_voice_{iid}")
    with c2:
        acting = st.selectbox("Acting instruction", ACTING_PRESETS, key=f"hume_acting_{iid}")

    custom_voice_id = st.text_input("Or paste custom voice ID", key=f"hume_custom_voice_{iid}",
                                     placeholder="e.g. abc123… (leave blank to use dropdown)")
    custom_acting = st.text_input("Or type custom acting instruction", key=f"hume_custom_{iid}",
                                   placeholder="e.g. 'frightened, rushed'")

    if not key:
        st.markdown(missing_key_html("HUME_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key=f"hume_gen_{iid}")
    with col_dl:
        dl_placeholder = st.empty()
    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True); return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True); return
        try:
            import requests, base64
            effective_acting = custom_acting.strip() or (None if acting == "None" else acting)
            headers = {"X-Hume-Api-Key": key, "Content-Type": "application/json"}
            utterance: dict = {"text": prompt}
            custom_vid = custom_voice_id.strip()
            if custom_vid:
                utterance["voice"] = {"id": custom_vid}
            elif voice_label != "Auto-generate (no voice)":
                utterance["voice"] = {"name": voice_label, "provider": "HUME_AI"}
            if effective_acting:
                utterance["description"] = effective_acting
            with st.spinner("Generating…"):
                resp = requests.post("https://api.hume.ai/v0/tts", json={"utterances": [utterance]},
                                     headers=headers, timeout=30)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ {resp.status_code}: {resp.text[:300]}</span>', unsafe_allow_html=True); return
            audio_b64 = resp.json()["generations"][0]["audio"]
            st.session_state.audio_store[store_key] = mp3_to_wav_fallback(base64.b64decode(audio_b64))
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True); traceback.print_exc()

    if store_key in st.session_state.audio_store:
        wav = st.session_state.audio_store[store_key]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, f"hume{_phase}_{iid}.wav", "audio/wav", key=f"hume_dl_{iid}")


def render_google_gemini(prompt: str, iid: int = 0, phase: str = ""):
    key = get_secret('GOOGLE_API_KEY')
    store_key = f"Google Gemini_{iid}"
    _phase = f"_{phase}" if phase.strip() else ""

    VOICES = ["Aoede", "Charon", "Fenrir", "Kore", "Leda", "Orus", "Puck", "Schedar",
              "Sulafat", "Umbriel", "Zephyr", "Achird", "Algieba", "Alnilam", "Autonoe",
              "Callirrhoe", "Despina", "Enceladus", "Erinome", "Gacrux", "Iocaste",
              "Laomedeia", "Pulcherrima", "Rasalgethi", "Sadachbia", "Sadaltager",
              "Vindemiatrix", "Zubenelgenubi"]
    STYLE_PRESETS = ["None (no style prompt)", "Say in an excited, upbeat tone",
                     "Say calmly and slowly", "Say in a spooky whisper",
                     "Say with sadness and empathy", "Say enthusiastically like a sports announcer",
                     "Say in a professional newscast style", "Say warmly, like speaking to a friend",
                     "Say dramatically, like a movie trailer"]

    c1, c2 = st.columns(2)
    with c1:
        voice = st.selectbox("Voice", VOICES, key=f"gg_voice_{iid}")
    with c2:
        style_preset = st.selectbox("Style prompt", STYLE_PRESETS, key=f"gg_style_{iid}")
    custom_style = st.text_input("Or type custom style prompt", key=f"gg_custom_{iid}",
                                  placeholder='e.g. "Say thoughtfully, with long pauses"')

    if not key:
        st.markdown(missing_key_html("GOOGLE_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key=f"gg_gen_{iid}")
    with col_dl:
        dl_placeholder = st.empty()
    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True); return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True); return
        try:
            import requests, base64
            eff_style = custom_style.strip() or (None if style_preset == "None (no style prompt)" else style_preset)
            full_text = f"{eff_style}: {prompt}" if eff_style else prompt
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={key}"
            body = {"contents": [{"parts": [{"text": full_text}]}],
                    "generationConfig": {"responseModalities": ["AUDIO"],
                                         "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
            with st.spinner("Generating…"):
                resp = requests.post(url, json=body, timeout=60)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ {resp.status_code}: {resp.text[:300]}</span>', unsafe_allow_html=True); return
            audio_b64 = resp.json()["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
            st.session_state.audio_store[store_key] = pcm_to_wav(base64.b64decode(audio_b64))
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True); traceback.print_exc()

    if store_key in st.session_state.audio_store:
        wav = st.session_state.audio_store[store_key]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, f"google_gemini{_phase}_{iid}.wav", "audio/wav", key=f"gg_dl_{iid}")


def render_azure(prompt: str, iid: int = 0, phase: str = ""):
    key = get_secret("AZURE_SPEECH_KEY")
    region = get_secret("AZURE_SPEECH_REGION") or "westus"
    store_key = f"Azure_{iid}"
    _phase = f"_{phase}" if phase.strip() else ""

    VOICES = {
        "Jenny (Female, en-US)":    ("en-US-JennyNeural", "en-US"),
        "Guy (Male, en-US)":        ("en-US-GuyNeural",   "en-US"),
        "Aria (Female, en-US)":     ("en-US-AriaNeural",  "en-US"),
        "Davis (Male, en-US)":      ("en-US-DavisNeural", "en-US"),
        "Jane (Female, en-US)":     ("en-US-JaneNeural",  "en-US"),
        "Jason (Male, en-US)":      ("en-US-JasonNeural", "en-US"),
        "Sara (Female, en-US)":     ("en-US-SaraNeural",  "en-US"),
        "Tony (Male, en-US)":       ("en-US-TonyNeural",  "en-US"),
        "Ryan (Male, en-GB)":       ("en-GB-RyanNeural",  "en-GB"),
        "Sonia (Female, en-GB)":    ("en-GB-SoniaNeural", "en-GB"),
    }
    STYLES = ["general", "cheerful", "sad", "angry", "fearful", "disgruntled", "serious",
              "affectionate", "gentle", "embarrassed", "empathetic", "newscast",
              "customerservice", "whispering", "shouting", "lyrical"]

    c1, c2 = st.columns(2)
    with c1:
        voice_label = st.selectbox("Voice", list(VOICES.keys()), key=f"az_voice_{iid}")
    with c2:
        style = st.selectbox("Speaking style", STYLES, key=f"az_style_{iid}")
    style_degree = st.slider("Style intensity", 0.1, 2.0, 1.0, 0.1, key=f"az_degree_{iid}")

    if not key:
        st.markdown(missing_key_html("AZURE_SPEECH_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key=f"az_gen_{iid}")
    with col_dl:
        dl_placeholder = st.empty()
    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True); return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True); return
        try:
            import requests
            voice_name, lang = VOICES[voice_label]
            ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis'
  xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='{lang}'>
  <voice name='{voice_name}'>
    <mstts:express-as style='{style}' styledegree='{style_degree:.1f}'>{prompt}</mstts:express-as>
  </voice>
</speak>"""
            tok_resp = requests.post(f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken",
                                      headers={"Ocp-Apim-Subscription-Key": key}, timeout=10)
            if tok_resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ Token error {tok_resp.status_code}</span>', unsafe_allow_html=True); return
            tts_url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
            headers = {"Authorization": f"Bearer {tok_resp.text}", "Content-Type": "application/ssml+xml",
                       "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm", "User-Agent": "TTS-Studio"}
            with st.spinner("Synthesizing…"):
                resp = requests.post(tts_url, data=ssml.encode("utf-8"), headers=headers, timeout=30)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ {resp.status_code}: {resp.text[:200]}</span>', unsafe_allow_html=True); return
            st.session_state.audio_store[store_key] = resp.content
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True); traceback.print_exc()

    if store_key in st.session_state.audio_store:
        wav = st.session_state.audio_store[store_key]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, f"azure{_phase}_{iid}.wav", "audio/wav", key=f"az_dl_{iid}")


def render_cartesia(prompt: str, iid: int = 0, phase: str = ""):
    key = get_secret('CARTESIA_API_KEY')
    store_key = f"Cartesia_{iid}"
    _phase = f"_{phase}" if phase.strip() else ""

    VOICES = {
        "Barbershop Man":   "a0e99841-438c-4a64-b679-ae501e7d6091",
        "Female Nurse":     "5c42302c-194b-4d0c-ba1a-8cb485c84ab9",
        "Sweet Lady":       "e3827ec5-697a-4b7c-9704-1a23041bbc51",
        "Friendly Reading": "79f8b5fb-2cc8-479a-80df-29f7a7cf1a3e",
        "Deep Male":        "ee7ea9f8-c0c1-498c-9279-764d6b56d189",
        "Calm Lady":        "b7d50908-b17c-442d-ad8d-810c63997ed9",
        "British Reporter": "71a7ad14-091d-4441-9c30-be5fc3e25d32",
    }
    EMOTIONS = ["neutral", "angry", "excited", "content", "sad", "scared", "happy",
                "enthusiastic", "elated", "euphoric", "triumphant", "surprised", "curious",
                "calm", "grateful", "sympathetic", "sarcastic", "dejected", "melancholic"]
    SPEEDS = {"Slowest": 0.6, "Slow": 0.8, "Normal": 1.0, "Fast": 1.2, "Fastest": 1.5}

    c1, c2, c3 = st.columns(3)
    with c1:
        voice_label = st.selectbox("Voice", list(VOICES.keys()), key=f"ca_voice_{iid}")
    with c2:
        emotion = st.selectbox("Emotion", EMOTIONS, key=f"ca_emotion_{iid}")
    with c3:
        speed_label = st.selectbox("Speed", list(SPEEDS.keys()), index=2, key=f"ca_speed_{iid}")

    if not key:
        st.markdown(missing_key_html("CARTESIA_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key=f"ca_gen_{iid}")
    with col_dl:
        dl_placeholder = st.empty()
    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True); return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True); return
        try:
            import requests
            body = {"model_id": "sonic-3", "transcript": prompt,
                    "voice": {"mode": "id", "id": VOICES[voice_label]},
                    "output_format": {"container": "wav", "encoding": "pcm_f32le", "sample_rate": 44100},
                    "generation_config": {"speed": SPEEDS[speed_label], "emotion": emotion}}
            with st.spinner("Generating…"):
                resp = requests.post("https://api.cartesia.ai/tts/bytes", json=body,
                                      headers={"X-API-Key": key, "Cartesia-Version": "2025-04-16",
                                               "Content-Type": "application/json"}, timeout=30)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ {resp.status_code}: {resp.text[:300]}</span>', unsafe_allow_html=True); return
            st.session_state.audio_store[store_key] = resp.content
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True); traceback.print_exc()

    if store_key in st.session_state.audio_store:
        wav = st.session_state.audio_store[store_key]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, f"cartesia{_phase}_{iid}.wav", "audio/wav", key=f"ca_dl_{iid}")


def render_deepgram(prompt: str, iid: int = 0, phase: str = ""):
    key = get_secret('DEEPGRAM_API_KEY')
    store_key = f"Deepgram_{iid}"
    _phase = f"_{phase}" if phase.strip() else ""

    VOICES = ["aura-2-thalia-en", "aura-2-andromeda-en", "aura-2-helena-en",
              "aura-2-apollo-en", "aura-2-orion-en", "aura-2-zeus-en",
              "aura-2-luna-en", "aura-2-stella-en", "aura-2-asteria-en",
              "aura-asteria-en", "aura-luna-en", "aura-stella-en",
              "aura-athena-en", "aura-hera-en", "aura-orion-en",
              "aura-arcas-en", "aura-perseus-en", "aura-angus-en",
              "aura-orpheus-en", "aura-helios-en", "aura-zeus-en"]
    ENCODING_OPTIONS = {"WAV/PCM (linear16)": "linear16", "MP3": "mp3", "Opus": "opus"}

    c1, c2 = st.columns(2)
    with c1:
        voice = st.selectbox("Voice model", VOICES, key=f"dg_voice_{iid}")
    with c2:
        enc_label = st.selectbox("Format", list(ENCODING_OPTIONS.keys()), key=f"dg_enc_{iid}")

    if not key:
        st.markdown(missing_key_html("DEEPGRAM_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key=f"dg_gen_{iid}")
    with col_dl:
        dl_placeholder = st.empty()
    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True); return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True); return
        try:
            import requests
            encoding = ENCODING_OPTIONS[enc_label]
            params = {"model": voice, "encoding": encoding}
            if encoding == "linear16":
                params["sample_rate"] = "24000"; params["container"] = "wav"
            with st.spinner("Generating…"):
                resp = requests.post("https://api.deepgram.com/v1/speak", params=params,
                                      json={"text": prompt},
                                      headers={"Authorization": f"Token {key}", "Content-Type": "application/json"},
                                      timeout=30)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ {resp.status_code}: {resp.text[:300]}</span>', unsafe_allow_html=True); return
            audio_bytes = resp.content
            if encoding == "mp3":
                audio_bytes = mp3_to_wav_fallback(audio_bytes)
            st.session_state.audio_store[store_key] = audio_bytes
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True); traceback.print_exc()

    if store_key in st.session_state.audio_store:
        wav = st.session_state.audio_store[store_key]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, f"deepgram{_phase}_{iid}.wav", "audio/wav", key=f"dg_dl_{iid}")


def render_fish_audio(prompt: str, iid: int = 0, phase: str = ""):
    key = get_secret('FISH_AUDIO_API_KEY')
    store_key = f"Fish Audio_{iid}"
    _phase = f"_{phase}" if phase.strip() else ""

    VOICES = {"Default (no reference)": None, "Custom reference ID…": "__custom__"}
    EMOTION_TAGS = ["None", "[laughing]", "[chuckling]", "[sobbing]", "[crying loudly]",
                    "[sighing]", "[panting]", "[groaning]", "[whispers]",
                    "[excited]", "[angry]", "[sad]", "[happy]",
                    "[super happy]", "[long pause]", "[pause]"]

    c1, c2 = st.columns(2)
    with c1:
        voice_label = st.selectbox("Voice preset", list(VOICES.keys()), key=f"fa_voice_{iid}")
    with c2:
        emotion_tag = st.selectbox("Inline emotion tag", EMOTION_TAGS, key=f"fa_emotion_{iid}")

    custom_ref = ""
    if voice_label == "Custom reference ID…":
        custom_ref = st.text_input("Voice reference ID (from fish.audio)", key=f"fa_ref_id_{iid}",
                                    placeholder="e.g. 802e3bc2b27e49c2995d23ef70e6ac89")

    if not key:
        st.markdown(missing_key_html("FISH_AUDIO_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key=f"fa_gen_{iid}")
    with col_dl:
        dl_placeholder = st.empty()
    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True); return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True); return
        try:
            import requests
            tag = "" if emotion_tag == "None" else emotion_tag + " "
            ref_id = custom_ref.strip() if voice_label == "Custom reference ID…" and custom_ref.strip() else None
            payload = {"text": tag + prompt, "format": "wav", "latency": "balanced"}
            if ref_id:
                payload["reference_id"] = ref_id
            with st.spinner("Generating…"):
                resp = requests.post("https://api.fish.audio/v1/tts", json=payload,
                                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                      timeout=60)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ {resp.status_code}: {resp.text[:300]}</span>', unsafe_allow_html=True); return
            st.session_state.audio_store[store_key] = resp.content
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True); traceback.print_exc()

    if store_key in st.session_state.audio_store:
        wav = st.session_state.audio_store[store_key]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, f"fish_audio{_phase}_{iid}.wav", "audio/wav", key=f"fa_dl_{iid}")


def render_neuphonic(prompt: str, iid: int = 0, phase: str = ""):
    key = get_secret('NEUPHONIC_API_KEY')
    store_key = f"Neuphonic_{iid}"
    _phase = f"_{phase}" if phase.strip() else ""

    LANGUAGES = {"English (US)": "en", "English (UK)": "en-gb", "Spanish": "es",
                 "French": "fr", "German": "de", "Portuguese": "pt",
                 "Italian": "it", "Dutch": "nl", "Polish": "pl", "Arabic": "ar"}

    c1, c2 = st.columns(2)
    with c1:
        lang_label = st.selectbox("Language", list(LANGUAGES.keys()), key=f"neu_lang_{iid}")
    with c2:
        voice_id_input = st.text_input("Voice ID (leave blank for default)", key=f"neu_voice_id_{iid}",
                                        placeholder="e.g. 8e9c4bc8-3979-48ab-8626-df53befc2090")
    speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.05, key=f"neu_speed_{iid}")

    if not key:
        st.markdown(missing_key_html("NEUPHONIC_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key=f"neu_gen_{iid}")
    with col_dl:
        dl_placeholder = st.empty()
    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True); return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True); return
        try:
            from pyneuphonic import Neuphonic, TTSConfig
        except ImportError:
            status.markdown('<span class="status-err">❌ pyneuphonic not installed. Run: pip install pyneuphonic</span>', unsafe_allow_html=True); return
        try:
            client = Neuphonic(api_key=key)
            sse = client.tts.SSEClient()
            tts_config = TTSConfig(speed=speed, lang_code=LANGUAGES[lang_label],
                                   voice_id=voice_id_input.strip() or None, sampling_rate=22050)
            with st.spinner("Generating…"):
                response = sse.send(prompt, tts_config=tts_config)
            pcm_chunks = [bytes(getattr(getattr(c, "data", None), "audio", None) or b"") for c in response]
            pcm_bytes = b"".join(pcm_chunks)
            if not pcm_bytes:
                status.markdown('<span class="status-err">❌ No audio returned</span>', unsafe_allow_html=True); return
            st.session_state.audio_store[store_key] = pcm_to_wav(pcm_bytes, sample_rate=22050)
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True); traceback.print_exc()

    if store_key in st.session_state.audio_store:
        wav = st.session_state.audio_store[store_key]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, f"neuphonic{_phase}_{iid}.wav", "audio/wav", key=f"neu_dl_{iid}")


def render_inworld(prompt: str, iid: int = 0, phase: str = ""):
    key = get_secret('INWORLD_API_KEY')
    store_key = f"Inworld_{iid}"
    _phase = f"_{phase}" if phase.strip() else ""

    st.info("Inworld TTS uses a character-based voice model. Provide workspace ID and character name below.")
    c1, c2 = st.columns(2)
    with c1:
        workspace_id = st.text_input("Workspace ID", value=get_secret("INWORLD_WORKSPACE_ID") or "",
                                      key=f"iw_workspace_{iid}", placeholder="workspaces/my-workspace")
    with c2:
        character_name = st.text_input("Character name", value=get_secret("INWORLD_CHARACTER") or "",
                                        key=f"iw_character_{iid}", placeholder="characters/my-character")

    if not key:
        st.markdown(missing_key_html("INWORLD_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key=f"iw_gen_{iid}")
    with col_dl:
        dl_placeholder = st.empty()
    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True); return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True); return
        try:
            import requests, base64
            with st.spinner("Generating…"):
                resp = requests.post("https://studio.inworld.ai/v1/tts:synthesize",
                                      json={"text": prompt, "character": character_name or "characters/default"},
                                      headers={"Authorization": f"Basic {key}", "Content-Type": "application/json"},
                                      timeout=30)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ {resp.status_code}: {resp.text[:300]}</span>', unsafe_allow_html=True); return
            audio_bytes = base64.b64decode(resp.json().get("audioContent", ""))
            st.session_state.audio_store[store_key] = mp3_to_wav_fallback(audio_bytes)
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True); traceback.print_exc()

    if store_key in st.session_state.audio_store:
        wav = st.session_state.audio_store[store_key]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, f"inworld{_phase}_{iid}.wav", "audio/wav", key=f"iw_dl_{iid}")


def render_async_ai(prompt: str, iid: int = 0, phase: str = ""):
    key = get_secret('ASYNC_API_KEY')
    store_key = f"Async AI_{iid}"
    _phase = f"_{phase}" if phase.strip() else ""

    MODELS = {"Async Flash v1 (fast)": "async_flash_v1.0", "Async v1 (quality)": "async_v1.0"}
    OUTPUT_FORMATS = {
        "WAV / PCM float32 (44.1 kHz)": {"container": "wav", "encoding": "pcm_f32le", "sample_rate": 44100},
        "WAV / PCM int16 (24 kHz)":     {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000},
        "MP3":                           {"container": "mp3", "encoding": "mp3", "sample_rate": 44100},
    }

    st.caption("Async uses voice UUIDs. Find yours at [console.async.com](https://console.async.com).")
    c1, c2 = st.columns(2)
    with c1:
        model_label = st.selectbox("Model", list(MODELS.keys()), key=f"async_model_{iid}")
    with c2:
        fmt_label = st.selectbox("Output format", list(OUTPUT_FORMATS.keys()), key=f"async_fmt_{iid}")
    voice_id = st.text_input("Voice UUID (from Async console)", key=f"async_voice_id_{iid}",
                              placeholder="e.g. e0f39dc4-f691-4e78-bba5-5c636692cc04")

    if not key:
        st.markdown(missing_key_html("ASYNC_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key=f"async_gen_{iid}")
    with col_dl:
        dl_placeholder = st.empty()
    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True); return
        if not voice_id.strip():
            status.markdown('<span class="status-warn">⚠️ Please enter a voice UUID</span>', unsafe_allow_html=True); return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True); return
        try:
            import requests
            body = {"model_id": MODELS[model_label], "transcript": prompt,
                    "voice": {"mode": "id", "id": voice_id.strip()},
                    "output_format": OUTPUT_FORMATS[fmt_label]}
            with st.spinner("Generating…"):
                resp = requests.post("https://api.async.com/text_to_speech", json=body,
                                      headers={"x-api-key": key, "version": "v1", "Content-Type": "application/json"},
                                      timeout=30)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ {resp.status_code}: {resp.text[:300]}</span>', unsafe_allow_html=True); return
            st.session_state.audio_store[store_key] = resp.content
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True); traceback.print_exc()

    if store_key in st.session_state.audio_store:
        wav = st.session_state.audio_store[store_key]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, f"async_ai{_phase}_{iid}.wav", "audio/wav", key=f"async_dl_{iid}")


def render_openai_tts(prompt: str, iid: int = 0, phase: str = ""):
    key = get_secret("OPENAI_API_KEY")
    store_key = f"OpenAI TTS_{iid}"
    _phase = f"_{phase}" if phase.strip() else ""

    VOICES = ["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]
    MODELS = {"gpt-4o-mini-tts (best quality)": "gpt-4o-mini-tts",
              "tts-1 (fast)": "tts-1", "tts-1-hd (high def)": "tts-1-hd"}
    STYLE_PRESETS = ["None", "Speak in a cheerful and positive tone.",
                     "Speak slowly and calmly, like a meditation guide.",
                     "Speak with excitement and energy.", "Speak in a sad, somber tone.",
                     "Speak authoritatively and confidently.", "Speak in a whisper.",
                     "Speak sarcastically.", "Speak warmly, like talking to a friend."]

    c1, c2 = st.columns(2)
    with c1:
        model_label = st.selectbox("Model", list(MODELS.keys()), key=f"oai_model_{iid}")
    with c2:
        voice = st.selectbox("Voice", VOICES, key=f"oai_voice_{iid}")
    style_preset = st.selectbox("Style instructions", STYLE_PRESETS, key=f"oai_style_{iid}")
    custom_instructions = st.text_input("Or type custom instructions (overrides preset)",
                                         key=f"oai_instructions_{iid}",
                                         placeholder='e.g. "Speak like a pirate, with a jolly tone."')

    if not key:
        st.markdown(missing_key_html("OPENAI_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key=f"oai_gen_{iid}")
    with col_dl:
        dl_placeholder = st.empty()
    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True); return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True); return
        try:
            import requests
            instructions = custom_instructions.strip() or (None if style_preset == "None" else style_preset)
            body = {"model": MODELS[model_label], "input": prompt, "voice": voice, "response_format": "wav"}
            if instructions:
                body["instructions"] = instructions
            with st.spinner("Generating…"):
                resp = requests.post("https://api.openai.com/v1/audio/speech", json=body,
                                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                      timeout=30)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ {resp.status_code}: {resp.text[:300]}</span>', unsafe_allow_html=True); return
            st.session_state.audio_store[store_key] = resp.content
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True); traceback.print_exc()

    if store_key in st.session_state.audio_store:
        wav = st.session_state.audio_store[store_key]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, f"openai_tts{_phase}_{iid}.wav", "audio/wav", key=f"oai_dl_{iid}")


def render_murf(prompt: str, iid: int = 0, phase: str = ""):
    key = get_secret("MURF_API_KEY")
    store_key = f"Murf AI_{iid}"
    _phase = f"_{phase}" if phase.strip() else ""

    VOICES = {"Natalie (Female, en-US)": "en-US-natalie", "Miles (Male, en-US)": "en-US-miles",
              "Ruby (Female, en-GB)": "en-UK-ruby", "Finley (Male, en-GB)": "en-UK-finley",
              "Aurora (Female, en-AU)": "en-AU-evelyn", "Aarav (Male, en-IN)": "en-IN-aarav",
              "Isabelle (Female, fr-FR)": "fr-FR-maxime", "Carlos (Male, es-ES)": "es-ES-carlos"}
    STYLES = ["Conversational", "Newscast", "Promo", "Narration", "Inspirational",
              "Calm", "Sad", "Angry", "Fearful", "Cheerful", "Empathetic", "Excited"]

    c1, c2 = st.columns(2)
    with c1:
        voice_label = st.selectbox("Voice", list(VOICES.keys()), key=f"murf_voice_{iid}")
    with c2:
        style = st.selectbox("Speaking style", STYLES, key=f"murf_style_{iid}")
    speed = st.slider("Speaking rate (%)", -50, 50, 0, 5, key=f"murf_rate_{iid}",
                      help="-50 = slowest, 0 = normal, +50 = fastest")

    if not key:
        st.markdown(missing_key_html("MURF_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key=f"murf_gen_{iid}")
    with col_dl:
        dl_placeholder = st.empty()
    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True); return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True); return
        try:
            import requests, base64
            body = {"voiceId": VOICES[voice_label], "style": style, "text": prompt,
                    "rate": speed, "format": "WAV", "encodeAsBase64": True}
            with st.spinner("Generating…"):
                resp = requests.post("https://api.murf.ai/v1/speech/generate", json=body,
                                      headers={"api-key": key, "Content-Type": "application/json"}, timeout=30)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ {resp.status_code}: {resp.text[:300]}</span>', unsafe_allow_html=True); return
            data = resp.json()
            audio_b64 = data.get("encodedAudio", "")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
            else:
                audio_url = data.get("audioFile") or data.get("audioUrl", "")
                if not audio_url:
                    status.markdown('<span class="status-err">❌ No audio in response</span>', unsafe_allow_html=True); return
                audio_bytes = requests.get(audio_url, timeout=20).content
            st.session_state.audio_store[store_key] = audio_bytes
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True); traceback.print_exc()

    if store_key in st.session_state.audio_store:
        wav = st.session_state.audio_store[store_key]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, f"murf_ai{_phase}_{iid}.wav", "audio/wav", key=f"murf_dl_{iid}")


def render_lmnt(prompt: str, iid: int = 0, phase: str = ""):
    key = get_secret("LMNT_API_KEY")
    store_key = f"LMNT_{iid}"
    _phase = f"_{phase}" if phase.strip() else ""

    VOICES = ["leah", "lily", "zoe", "luna", "nova", "ava", "aria", "emma", "olivia",
              "drew", "brandon", "miles", "adam"]
    MODELS = {"Aurora (latest)": "aurora", "Blizzard (legacy)": "blizzard"}

    c1, c2 = st.columns(2)
    with c1:
        voice = st.selectbox("Voice", VOICES, key=f"lmnt_voice_{iid}")
    with c2:
        model_label = st.selectbox("Model", list(MODELS.keys()), key=f"lmnt_model_{iid}")
    speed = st.slider("Speed", 0.25, 2.0, 1.0, 0.05, key=f"lmnt_speed_{iid}")

    if not key:
        st.markdown(missing_key_html("LMNT_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key=f"lmnt_gen_{iid}")
    with col_dl:
        dl_placeholder = st.empty()
    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True); return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True); return
        try:
            import requests
            body = {"text": prompt, "voice": voice, "model": MODELS[model_label], "format": "wav", "speed": speed}
            with st.spinner("Generating…"):
                resp = requests.post("https://api.lmnt.com/v1/ai/speech/bytes", json=body,
                                      headers={"X-API-Key": key, "Content-Type": "application/json"}, timeout=30)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ {resp.status_code}: {resp.text[:300]}</span>', unsafe_allow_html=True); return
            st.session_state.audio_store[store_key] = resp.content
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True); traceback.print_exc()

    if store_key in st.session_state.audio_store:
        wav = st.session_state.audio_store[store_key]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, f"lmnt{_phase}_{iid}.wav", "audio/wav", key=f"lmnt_dl_{iid}")


def render_rime(prompt: str, iid: int = 0, phase: str = ""):
    key = get_secret("RIME_API_KEY")
    store_key = f"Rime_{iid}"
    _phase = f"_{phase}" if phase.strip() else ""

    MODELS = {"Mistv2 (fast, business)": "mistv2", "Arcana (expressive)": "arcana", "Mist (legacy)": "mist"}
    VOICES = ["river", "cove", "luna", "joy", "juan", "grace", "maya", "alex",
              "sam", "morgan", "casey", "riley", "quinn", "sage", "sky"]
    LANGUAGES = {"English": "eng", "Spanish": "spa", "French": "fra", "German": "deu", "Portuguese": "por"}

    c1, c2, c3 = st.columns(3)
    with c1:
        model_label = st.selectbox("Model", list(MODELS.keys()), key=f"rime_model_{iid}")
    with c2:
        voice = st.selectbox("Voice", VOICES, key=f"rime_voice_{iid}")
    with c3:
        lang_label = st.selectbox("Language", list(LANGUAGES.keys()), key=f"rime_lang_{iid}")

    if not key:
        st.markdown(missing_key_html("RIME_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key=f"rime_gen_{iid}")
    with col_dl:
        dl_placeholder = st.empty()
    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True); return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True); return
        try:
            import requests
            body = {"text": prompt, "speaker": voice, "modelId": MODELS[model_label],
                    "lang": LANGUAGES[lang_label], "audioFormat": "wav"}
            with st.spinner("Generating…"):
                resp = requests.post("https://users.rime.ai/v1/rime-tts", json=body,
                                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                               "Accept": "audio/wav"}, timeout=30)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ {resp.status_code}: {resp.text[:300]}</span>', unsafe_allow_html=True); return
            content = resp.content
            if content[:4] != b"RIFF":
                content = pcm_to_wav(content, sample_rate=22050)
            st.session_state.audio_store[store_key] = content
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True); traceback.print_exc()

    if store_key in st.session_state.audio_store:
        wav = st.session_state.audio_store[store_key]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, f"rime{_phase}_{iid}.wav", "audio/wav", key=f"rime_dl_{iid}")


def render_minimax(prompt: str, iid: int = 0, phase: str = ""):
    key  = get_secret("MINIMAX_API_KEY")
    gid  = get_secret("MINIMAX_GROUP_ID")
    store_key = f"MiniMax_{iid}"
    _phase = f"_{phase}" if phase.strip() else ""

    VOICES = ["Calm_Woman", "Energetic_Man", "Warm_Man", "Gentle_Woman", "Professional_Woman",
              "Friendly_Man", "Authoritative_Man", "Cheerful_Woman", "Narration_Man", "Podcast_Host"]
    EMOTIONS = ["auto", "happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]
    MODELS = {"speech-2.6-hd (best quality)": "speech-2.6-hd", "speech-2.6-turbo (fast)": "speech-2.6-turbo",
              "speech-02-hd": "speech-02-hd", "speech-02-turbo": "speech-02-turbo"}

    c1, c2 = st.columns(2)
    with c1:
        voice = st.selectbox("Voice", VOICES, key=f"mm_voice_{iid}")
    with c2:
        emotion = st.selectbox("Emotion", EMOTIONS, key=f"mm_emotion_{iid}")
    c3, c4 = st.columns(2)
    with c3:
        model_label = st.selectbox("Model", list(MODELS.keys()), key=f"mm_model_{iid}")
    with c4:
        speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.05, key=f"mm_speed_{iid}")
    group_id_input = st.text_input("Group ID (required)", value=gid or "", key=f"mm_gid_{iid}",
                                    placeholder="e.g. 123456789")

    if not key:
        st.markdown(missing_key_html("MINIMAX_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key=f"mm_gen_{iid}")
    with col_dl:
        dl_placeholder = st.empty()
    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True); return
        if not group_id_input.strip():
            status.markdown('<span class="status-warn">⚠️ Group ID is required</span>', unsafe_allow_html=True); return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True); return
        try:
            import requests, binascii
            voice_setting: dict = {"voice_id": voice, "speed": speed}
            if emotion != "auto":
                voice_setting["emotion"] = emotion
            body = {"model": MODELS[model_label], "text": prompt, "voice_setting": voice_setting,
                    "audio_setting": {"sample_rate": 24000, "bitrate": 128000, "format": "wav"}}
            with st.spinner("Generating…"):
                resp = requests.post(f"https://api.minimax.io/v1/t2a_v2?GroupId={group_id_input.strip()}",
                                      json=body, headers={"Authorization": f"Bearer {key}",
                                                          "Content-Type": "application/json"}, timeout=60)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ {resp.status_code}: {resp.text[:300]}</span>', unsafe_allow_html=True); return
            hex_audio = resp.json().get("data", {}).get("audio", "")
            if not hex_audio:
                status.markdown(f'<span class="status-err">❌ No audio in response</span>', unsafe_allow_html=True); return
            st.session_state.audio_store[store_key] = binascii.unhexlify(hex_audio)
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True); traceback.print_exc()

    if store_key in st.session_state.audio_store:
        wav = st.session_state.audio_store[store_key]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, f"minimax{_phase}_{iid}.wav", "audio/wav", key=f"mm_dl_{iid}")


def render_smallest_ai(prompt: str, iid: int = 0, phase: str = ""):
    key = get_secret("SMALLEST_AI_API_KEY")
    store_key = f"Smallest AI_{iid}"
    _phase = f"_{phase}" if phase.strip() else ""

    VOICES = ["emily", "aria", "jessica", "michael", "ethan", "luna", "zoe", "liam", "noah", "ava"]
    MODELS = {"Lightning v3 (recommended)": "lightning-v3", "Lightning v2": "lightning-v2"}

    c1, c2 = st.columns(2)
    with c1:
        voice = st.selectbox("Voice", VOICES, key=f"sai_voice_{iid}")
    with c2:
        model_label = st.selectbox("Model", list(MODELS.keys()), key=f"sai_model_{iid}")
    speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.05, key=f"sai_speed_{iid}")

    if not key:
        st.markdown(missing_key_html("SMALLEST_AI_API_KEY"), unsafe_allow_html=True)

    col_gen, col_dl = st.columns([1, 1])
    with col_gen:
        gen = st.button("▶ Generate", key=f"sai_gen_{iid}")
    with col_dl:
        dl_placeholder = st.empty()
    status = st.empty()
    audio_placeholder = st.empty()

    if gen:
        if not key:
            status.markdown('<span class="status-err">❌ No API key</span>', unsafe_allow_html=True); return
        if not prompt.strip():
            status.markdown('<span class="status-warn">⚠️ Prompt is empty</span>', unsafe_allow_html=True); return
        try:
            import requests
            body = {"text": prompt, "voice_id": voice, "model": MODELS[model_label],
                    "output_format": "wav", "speed": speed, "sample_rate": 24000}
            with st.spinner("Generating…"):
                resp = requests.post("https://waves-api.smallest.ai/api/v1/tts/get_speech", json=body,
                                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                      timeout=30)
            if resp.status_code != 200:
                status.markdown(f'<span class="status-err">❌ {resp.status_code}: {resp.text[:300]}</span>', unsafe_allow_html=True); return
            st.session_state.audio_store[store_key] = resp.content
            status.markdown('<span class="status-ok">✓ Generated</span>', unsafe_allow_html=True)
        except Exception as e:
            status.markdown(f'<span class="status-err">❌ {e}</span>', unsafe_allow_html=True); traceback.print_exc()

    if store_key in st.session_state.audio_store:
        wav = st.session_state.audio_store[store_key]
        audio_placeholder.audio(wav, format="audio/wav")
        dl_placeholder.download_button("⬇ Download .wav", wav, f"smallest_ai{_phase}_{iid}.wav", "audio/wav", key=f"sai_dl_{iid}")


# ══════════════════════════════════════════════════════════════════════════════
# Provider registry
# ══════════════════════════════════════════════════════════════════════════════
PROVIDERS = [
    {"name": "ElevenLabs",        "key": "el",    "badge": "Free + Paid", "badge_class": "badge-free", "emoji": "🎧",
     "desc": "Industry-leading expressive TTS. Supports audio emotion tags like [laughs], [whispers]. Uses Turbo v2.5 model.",
     "render": render_elevenlabs},
    {"name": "Hume (Octave)",     "key": "hume",  "badge": "Free + Paid", "badge_class": "badge-free", "emoji": "🧠",
     "desc": "LLM-based TTS that understands emotional context. Supports acting instructions like 'frightened, rushed'.",
     "render": render_hume},
    {"name": "Google Gemini TTS", "key": "gg",    "badge": "Free tier",   "badge_class": "badge-free", "emoji": "✨",
     "desc": "Gemini 2.5 Flash TTS. Natural-language style prompts control tone, accent, pacing and emotion.",
     "render": render_google_gemini},
    {"name": "Microsoft Azure Speech", "key": "az", "badge": "Free tier", "badge_class": "badge-free", "emoji": "☁️",
     "desc": "Neural TTS with SSML emotion styles (cheerful, sad, whispering, etc.) and style degree control.",
     "render": render_azure},
    {"name": "Cartesia Sonic 3",  "key": "ca",    "badge": "Free tier",   "badge_class": "badge-free", "emoji": "⚡",
     "desc": "Ultra-low latency (~40ms TTFA). Rich emotion palette (60+ emotions) and speed/volume controls.",
     "render": render_cartesia},
    {"name": "Deepgram Aura",     "key": "dg",    "badge": "$200 free credit", "badge_class": "badge-paid", "emoji": "🔊",
     "desc": "Enterprise-grade TTS (Aura-2). Built for real-time voice agents. 40+ voices with domain-accurate pronunciation.",
     "render": render_deepgram},
    {"name": "Fish Audio",        "key": "fa",    "badge": "Free tier",   "badge_class": "badge-free", "emoji": "🐟",
     "desc": "Multilingual TTS with inline emotion tags ([laughing], [whispers], [excited]). Excellent for Asian languages.",
     "render": render_fish_audio},
    {"name": "Neuphonic",         "key": "neu",   "badge": "Free beta",   "badge_class": "badge-free", "emoji": "🌊",
     "desc": "Free beta TTS with speed and pitch controls. Multiple languages supported.",
     "render": render_neuphonic},
    {"name": "Inworld AI",        "key": "iw",    "badge": "Free credits","badge_class": "badge-free", "emoji": "🎮",
     "desc": "Character-based TTS for games and interactive experiences. Requires workspace + character setup.",
     "render": render_inworld},
    {"name": "Async AI",          "key": "async", "badge": "Free tier",   "badge_class": "badge-free", "emoji": "🔄",
     "desc": "OpenAI-compatible TTS API with speed control. Drop-in replacement for OpenAI TTS.",
     "render": render_async_ai},
    {"name": "OpenAI TTS",        "key": "oai",   "badge": "Paid",        "badge_class": "badge-paid", "emoji": "🤖",
     "desc": "gpt-4o-mini-tts with natural-language style instructions. 13 voices, 50+ languages. Simple and fast.",
     "render": render_openai_tts},
    {"name": "Murf AI",           "key": "murf",  "badge": "Paid",        "badge_class": "badge-paid", "emoji": "🎤",
     "desc": "Studio-grade TTS with 150+ voices across 20+ languages and 20+ speaking styles.",
     "render": render_murf},
    {"name": "LMNT",              "key": "lmnt",  "badge": "Paid",        "badge_class": "badge-paid", "emoji": "⚡",
     "desc": "Ultra-low-latency TTS (<150ms). Aurora and Blizzard models, 24+ languages.",
     "render": render_lmnt},
    {"name": "Rime",              "key": "rime",  "badge": "Paid",        "badge_class": "badge-paid", "emoji": "🌊",
     "desc": "Trained on real conversations. Mist v2 (business) and Arcana (expressive) models. 300+ voices.",
     "render": render_rime},
    {"name": "MiniMax TTS",       "key": "mm",    "badge": "Paid",        "badge_class": "badge-paid", "emoji": "🧩",
     "desc": "Speech-2.6 HD/Turbo: 40+ languages, 300+ voices, emotion control, excellent Asian language support.",
     "render": render_minimax},
    {"name": "Smallest AI",       "key": "sai",   "badge": "Paid",        "badge_class": "badge-paid", "emoji": "⚡",
     "desc": "Lightning-fast TTS with sub-200ms latency. Simple REST API, competitive pricing. Lightning v3 model.",
     "render": render_smallest_ai},
]


# ══════════════════════════════════════════════════════════════════════════════
# Render all provider cards with multi-instance support
# ══════════════════════════════════════════════════════════════════════════════
for provider in PROVIDERS:
    pname = provider["name"]
    pkey  = provider["key"]
    _init_provider(pname)
    instance_ids = st.session_state.provider_instance_ids[pname]

    # ── Provider header card ───────────────────────────────────────────────────
    st.markdown(
        f"""<div class="provider-card">
          <div class="provider-header">
            <span style="font-size:1.3rem">{provider['emoji']}</span>
            <span class="provider-name">{pname}</span>
            <span class="provider-badge {provider['badge_class']}">{provider['badge']}</span>
          </div>
          <div style="font-size:.82rem;color:#6b7280;">{provider['desc']}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── One sub-card per instance ──────────────────────────────────────────────
    for idx, iid in enumerate(instance_ids):
        n = len(instance_ids)

        # Instance header row: label + optional remove button
        if n > 1:
            hdr_col, rm_col = st.columns([11, 1])
            with hdr_col:
                st.markdown(
                    f'<div class="instance-label">Voice {idx + 1}</div>',
                    unsafe_allow_html=True,
                )
            with rm_col:
                if st.button("✕", key=f"rm_{pkey}_{iid}", help="Remove this instance"):
                    _remove_instance(pname, iid)
                    st.rerun()
        else:
            st.markdown('<div class="instance-label">Voice 1</div>', unsafe_allow_html=True)

        # Render the provider's controls inside a light sub-card
        with st.container():
            st.markdown('<div class="instance-card">', unsafe_allow_html=True)
            try:
                provider["render"](prompt_text, iid, phase_emotion)
            except Exception as e:
                st.error(f"⚠️ Error in {pname}: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Add voice button ───────────────────────────────────────────────────────
    if st.button(f"＋ Add {pname} Voice", key=f"add_{pkey}", use_container_width=False):
        _add_instance(pname)
        st.rerun()

    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# Download All
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## ⬇ Download All Generated Samples")
available = {k: v for k, v in st.session_state.audio_store.items() if v}

if not available:
    st.info("Generate at least one sample above to enable bulk download.")
else:
    # Pretty labels for the summary
    labels = ", ".join(
        f"{k.rsplit('_', 1)[0]} #{int(k.rsplit('_', 1)[1]) + 1}"
        for k in sorted(available)
    )
    st.success(f"{len(available)} sample(s) ready: {labels}")

    def build_zip(audio_dict: dict, phase: str = "") -> bytes:
        _phase = f"_{phase}" if phase.strip() else ""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for store_key, wav_bytes in audio_dict.items():
                # store_key = "ElevenLabs_0", "ElevenLabs_1", etc.
                provider_part, idx_part = store_key.rsplit("_", 1)
                safe_name = provider_part.lower().replace(" ", "_").replace("/", "_")
                filename = f"{safe_name}{_phase}_voice{int(idx_part) + 1}.wav"
                zf.writestr(filename, wav_bytes)
        return buf.getvalue()

    zip_bytes = build_zip(available, phase_emotion)
    st.download_button(
        label=f"⬇ Download all {len(available)} samples as .zip",
        data=zip_bytes,
        file_name="tts_samples.zip",
        mime="application/zip",
        use_container_width=True,
        key="dl_all",
    )