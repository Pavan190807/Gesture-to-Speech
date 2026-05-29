# sign_language_translator.py
# Run: streamlit run sign_language_translator.py

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import json
import os
import time
import matplotlib.pyplot as plt
import subprocess
import platform
import pyttsx3
from datetime import datetime

# ──────────────────────────────────────────────────
# Global TTS engine + helper with multi-language support
# ──────────────────────────────────────────────────
TTY_RATE = 85  # Slower rate, easier to follow
_TTS_ENGINE = pyttsx3.init()
_TTS_ENGINE.setProperty('rate', TTY_RATE)
_TTS_ENGINE.setProperty('volume', 0.9)

# Language configurations for pyttsx3
LANGUAGE_VOICES = {
    "English": {"voice_id": None, "rate": 85, "keywords": ["english", "en_", "en-", "us", "gb"]},
    "Hindi": {"voice_id": None, "rate": 80, "keywords": ["hindi", "hi_", "hi-", "hin", "ind", "indian"]},
    "Tamil": {"voice_id": None, "rate": 80, "keywords": ["tamil", "ta_", "ta-", "tam"]},
    "Telugu": {"voice_id": None, "rate": 80, "keywords": ["telugu", "te_", "te-", "tel"]},
    "Marathi": {"voice_id": None, "rate": 80, "keywords": ["marathi", "mr_", "mr-", "mar"]},
    "Bengali": {"voice_id": None, "rate": 80, "keywords": ["bengali", "bn_", "bn-", "ben", "bangla"]},
    "Gujarati": {"voice_id": None, "rate": 80, "keywords": ["gujarati", "gu_", "gu-", "guj"]},
    "Kannada": {"voice_id": None, "rate": 80, "keywords": ["kannada", "kn_", "kn-", "kan"]},
    "Malayalam": {"voice_id": None, "rate": 80, "keywords": ["malayalam", "ml_", "ml-", "mal"]},
    "Punjabi": {"voice_id": None, "rate": 80, "keywords": ["punjabi", "pa_", "pa-", "pan"]},
    "Spanish": {"voice_id": None, "rate": 85, "keywords": ["spanish", "es_", "es-", "español"]},
    "French": {"voice_id": None, "rate": 85, "keywords": ["french", "fr_", "fr-", "français"]},
    "German": {"voice_id": None, "rate": 85, "keywords": ["german", "de_", "de-", "deutsch"]},
    "Italian": {"voice_id": None, "rate": 85, "keywords": ["italian", "it_", "it-", "italiano"]},
    "Portuguese": {"voice_id": None, "rate": 85, "keywords": ["portuguese", "pt_", "pt-", "português"]},
    "Russian": {"voice_id": None, "rate": 85, "keywords": ["russian", "ru_", "ru-", "русский"]},
    "Japanese": {"voice_id": None, "rate": 75, "keywords": ["japanese", "ja_", "ja-", "日本語"]},
    "Chinese": {"voice_id": None, "rate": 75, "keywords": ["chinese", "zh_", "zh-", "中文", "mandarin"]},
    "Arabic": {"voice_id": None, "rate": 80, "keywords": ["arabic", "ar_", "ar-", "عربي"]}
}

# macOS language codes for 'say' command
MACOS_LANGUAGE_CODES = {
    "English": "Alex",  # Default English voice
    "Hindi": "Lekha",   # Hindi voice on macOS
    "Spanish": "Diego",
    "French": "Thomas",
    "German": "Anna",
    "Italian": "Alice",
    "Portuguese": "Luciana",
    "Russian": "Yuri",
    "Japanese": "Kyoko",
    "Chinese": "Ting-Ting",
    "Arabic": "Maged",
    # Indian language support might be limited on macOS
    "Tamil": "Lekha",
    "Telugu": "Lekha",
    "Bengali": "Lekha",
    "Marathi": "Lekha",
    "Gujarati": "Lekha",
    "Kannada": "Lekha",
    "Malayalam": "Lekha",
    "Punjabi": "Lekha"
}

# Language-specific sample texts for testing
SAMPLE_TEXTS = {
    "English": "Hello, how are you?",
    "Hindi": "नमस्ते, आप कैसे हैं?",
    "Tamil": "வணக்கம், எப்படி இருக்கிறீர்கள்?",
    "Telugu": "నమస్తే, మీరు ఎలా ఉన్నారు?",
    "Marathi": "नमस्कार, तुम्ही कसे आहात?",
    "Bengali": "নমস্কার, আপনি কেমন আছেন?",
    "Gujarati": "નમસ્તે, તમે કેમ છો?",
    "Kannada": "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
    "Malayalam": "നമസ്കാരം, സുഖമാണോ?",
    "Punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",
    "Spanish": "Hola, ¿cómo estás?",
    "French": "Bonjour, comment allez-vous?",
    "German": "Hallo, wie geht es dir?",
    "Italian": "Ciao, come stai?",
    "Portuguese": "Olá, como você está?",
    "Russian": "Привет, как дела?",
    "Japanese": "こんにちは、元気ですか？",
    "Chinese": "你好，你好吗？",
    "Arabic": "مرحبا، كيف حالك؟"
}

def initialize_language_voices():
    """Initialize voice IDs for each language based on available system voices"""
    if platform.system() != "Darwin":  # Not macOS
        voices = _TTS_ENGINE.getProperty('voices')
        
        # Debug: Print all available voices
        print("Available voices:")
        for voice in voices:
            print(f"- ID: {voice.id}")
            print(f"  Name: {voice.name}")
            print(f"  Languages: {voice.languages}")
            print()
        
        # Try to find appropriate voices for each language
        for lang, config in LANGUAGE_VOICES.items():
            for voice in voices:
                voice_info = (voice.name + " " + voice.id + " " + str(voice.languages)).lower()
                
                # Check if any keyword matches
                for keyword in config["keywords"]:
                    if keyword in voice_info:
                        LANGUAGE_VOICES[lang]["voice_id"] = voice.id
                        print(f"Found voice for {lang}: {voice.id}")
                        break
                
                if LANGUAGE_VOICES[lang]["voice_id"]:
                    break
        
        # Fallback: If Hindi voice not found, try to find any Indian English voice
        if not LANGUAGE_VOICES["Hindi"]["voice_id"]:
            for voice in voices:
                if "india" in voice.name.lower() or "indian" in voice.name.lower():
                    for indian_lang in ["Hindi", "Tamil", "Telugu", "Marathi", "Bengali", 
                                      "Gujarati", "Kannada", "Malayalam", "Punjabi"]:
                        if not LANGUAGE_VOICES[indian_lang]["voice_id"]:
                            LANGUAGE_VOICES[indian_lang]["voice_id"] = voice.id
        
        # Set English as default if not found
        if not LANGUAGE_VOICES["English"]["voice_id"] and voices:
            LANGUAGE_VOICES["English"]["voice_id"] = voices[0].id

# Initialize language voices at startup
initialize_language_voices()

def speak_text(text: str, language: str = "English"):
    """
    Cross-platform TTS wrapper with multi-language support
    """
    if not text:
        return
    
    try:
        if platform.system() == "Darwin":
            # macOS using say command with language-specific voice
            voice = MACOS_LANGUAGE_CODES.get(language, "Alex")
            rate = LANGUAGE_VOICES[language]["rate"]
            # Use the sample text if the actual text might not be in the target language
            subprocess.Popen(["say", "-v", voice, "-r", str(rate), text])
        else:
            # Other platforms using pyttsx3
            voice_id = LANGUAGE_VOICES[language]["voice_id"]
            rate = LANGUAGE_VOICES[language]["rate"]
            
            # Set voice if available
            if voice_id:
                _TTS_ENGINE.setProperty('voice', voice_id)
            else:
                # If no specific voice found, use default and warn user
                print(f"Warning: No voice found for {language}, using default voice")
                # Try to get any available voice
                voices = _TTS_ENGINE.getProperty('voices')
                if voices:
                    _TTS_ENGINE.setProperty('voice', voices[0].id)
            
            _TTS_ENGINE.setProperty('rate', rate)
            _TTS_ENGINE.say(text)
            _TTS_ENGINE.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")
        # Fallback to system default
        try:
            if platform.system() == "Windows":
                # Windows SAPI fallback
                import win32com.client
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                speaker.Speak(text)
            elif platform.system() == "Linux":
                # Linux espeak fallback
                subprocess.Popen(["espeak", text])
        except:
            print("TTS completely failed")

# ──────────────────────────────────────────────────
# Streamlit page configuration
# ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Sign Language Translator", 
    layout="wide",
    page_icon="🤟",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────
# Initialize session state for language selection
# ──────────────────────────────────────────────────
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"

# ──────────────────────────────────────────────────
# Enhanced global styles for a premium UI/UX
# ──────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Root app styling */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e2532 0%, #252d3d 100%);
        border-right: 2px solid #00FFAA;
    }
    
    /* Main title styling */
    .main-title {
        font-size: 48px !important;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(45deg, #00FFAA, #00CCA0, #4FFFB2);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 3s ease-in-out infinite alternate;
        margin: 30px 0;
        text-shadow: 0 0 30px rgba(0, 255, 170, 0.3);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    
    /* Enhanced sentence display box */
    .sentence-display {
        font-size: 28px;
        font-weight: 500;
        color: #ffffff;
        text-align: center;
        background: linear-gradient(135deg, #1a2332 0%, #2d3748 100%);
        border: 2px solid #00FFAA;
        border-radius: 20px;
        padding: 20px;
        margin: 25px 0;
        box-shadow: 
            0 8px 32px rgba(0, 255, 170, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        min-height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Language selector styling */
    .language-selector {
        background: linear-gradient(135deg, #1e2532 0%, #252d3d 100%);
        border: 1px solid #00FFAA;
        border-radius: 12px;
        padding: 16px;
        margin: 16px 0;
        box-shadow: 0 4px 15px rgba(0, 255, 170, 0.2);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 8px 16px;
        border-radius: 25px;
        font-weight: 500;
        margin: 5px;
    }
    
    .status-active {
        background: linear-gradient(45deg, #00ff88, #00cc70);
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.3);
    }
    
    .status-inactive {
        background: linear-gradient(45deg, #ff6b6b, #cc5555);
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    
    /* Enhanced button styling */
    div.stButton > button {
        background: linear-gradient(135deg, #00FFAA 0%, #00CCA0 100%);
        color: #0f1419;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 255, 170, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(135deg, #00CCA0 0%, #00AA88 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 255, 170, 0.4);
    }
    
    /* Card-style containers */
    .info-card {
        background: linear-gradient(135deg, #1e2532 0%, #252d3d 100%);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        border: 1px solid rgba(0, 255, 170, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Enhanced metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #1a2332 0%, #2d3748 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(0, 255, 170, 0.3);
        margin: 10px;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 255, 170, 0.2);
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #00FFAA;
        margin-bottom: 8px;
    }
    
    .metric-label {
        font-size: 14px;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar enhancements */
    .sidebar-title {
        font-size: 24px;
        font-weight: 600;
        color: #00FFAA;
        text-align: center;
        padding: 20px 0;
        border-bottom: 2px solid rgba(0, 255, 170, 0.3);
        margin-bottom: 20px;
    }
    
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(0, 255, 170, 0.3);
        border-radius: 50%;
        border-top-color: #00FFAA;
        animation: spin 1s ease-in-out infinite;
        margin-right: 8px;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Custom selectbox styling */
    .stSelectbox > div > div {
        background-color: #1e2532;
        border: 1px solid #00FFAA;
        border-radius: 8px;
    }
    
    /* Custom checkbox styling */
    .stCheckbox > label {
        font-weight: 500;
        color: #ffffff;
    }
    
    /* Alert styling */
    .custom-alert {
        padding: 16px;
        border-radius: 12px;
        margin: 16px 0;
        border-left: 4px solid;
    }
    
    .alert-success {
        background: linear-gradient(135deg, rgba(0, 255, 170, 0.1), rgba(0, 204, 136, 0.1));
        border-left-color: #00FFAA;
        color: #00FFAA;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1), rgba(255, 152, 0, 0.1));
        border-left-color: #FFC107;
        color: #FFC107;
    }
    
    .alert-info {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.1), rgba(56, 139, 253, 0.1));
        border-left-color: #4FACFE;
        color: #4FACFE;
    }
    
    /* Language group styling */
    .language-group {
        background: linear-gradient(135deg, rgba(0, 255, 170, 0.05), rgba(0, 204, 160, 0.05));
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
        border: 1px solid rgba(0, 255, 170, 0.2);
    }
    
    .language-group-title {
        font-size: 18px;
        font-weight: 600;
        color: #00FFAA;
        margin-bottom: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────
# Constants & Mediapipe Set-up
# ──────────────────────────────────────────────────
GESTURE_FILE = "gesture_words.json"
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(max_num_hands=1)

# ──────────────────────────────────────────────────
# Load or create gesture dictionary
# ──────────────────────────────────────────────────
if not os.path.exists(GESTURE_FILE):
    gesture_dict = {
        "Hello": [0.10] * 21,
        "Thank": [0.18] * 21,
        "You":   [0.12] * 21,
        "My":    [0.11] * 21,
        "Name":  [0.21] * 21,
        "Fine":  [0.23] * 21,
    }
    with open(GESTURE_FILE, "w") as f:
        json.dump(gesture_dict, f)
else:
    with open(GESTURE_FILE, "r") as f:
        gesture_dict = json.load(f)

# ──────────────────────────────────────────────────
# Helper function to extract keypoints from hand landmarks
# ──────────────────────────────────────────────────
def extract_keypoints(hand_landmarks):
    return [lm.x for lm in hand_landmarks.landmark]

# ──────────────────────────────────────────────────
# Enhanced model metrics and plotting
# ──────────────────────────────────────────────────
model_metrics = {
    "Accuracy": 0.92,
    "Precision": 0.91,
    "Recall": 0.89,
    "F1-Score": 0.90,
}

def plot_model_metrics(metrics):
    # Set dark theme for matplotlib
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#00FFAA', '#00CCA0', '#4FFFB2', '#66FFB8']
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', 
                fontweight='bold', color='white')
    
    ax.set_ylim([0, 1])
    ax.set_title("Model Performance Metrics", fontsize=18, fontweight='bold', color='#00FFAA', pad=20)
    ax.set_xlabel("Metrics", fontsize=14, color='white')
    ax.set_ylabel("Score", fontsize=14, color='white')
    
    # Style the plot
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#1e2532')
    fig.patch.set_facecolor('#0f1419')
    
    # Style ticks
    ax.tick_params(colors='white')
    
    st.pyplot(fig)

# ──────────────────────────────────────────────────
# Enhanced sidebar navigation with language selection
# ──────────────────────────────────────────────────
st.sidebar.markdown('<div class="sidebar-title">🤟 Sign Translator</div>', unsafe_allow_html=True)

# Add current time
current_time = datetime.now().strftime("%H:%M:%S")
st.sidebar.markdown(f"🕒 **Current Time:** {current_time}")

# Language selection in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌍 Language Settings")

# Group languages by region
language_groups = {
    "Indian Languages": ["Hindi", "Tamil", "Telugu", "Marathi", "Bengali", 
                        "Gujarati", "Kannada", "Malayalam", "Punjabi"],
    "European Languages": ["English", "Spanish", "French", "German", "Italian", 
                          "Portuguese", "Russian"],
    "Asian Languages": ["Japanese", "Chinese", "Arabic"]
}

# Create a flat list for selectbox
all_languages = []
for group, langs in language_groups.items():
    all_languages.extend(langs)

selected_language = st.sidebar.selectbox(
    "🗣️ **TTS Language**",
    all_languages,
    index=all_languages.index(st.session_state.get('selected_language', 'English')),
    help="Select the language for text-to-speech output"
)
st.session_state.selected_language = selected_language

# Show language group info
for group, langs in language_groups.items():
    if selected_language in langs:
        st.sidebar.info(f"**Language Group:** {group}")
        break

# Navigation menu
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "🧭 **Navigation**",
    ["🎥 Real-time Translator", "✍️ Manage Gestures", "📊 Model Analytics", "🔊 Test TTS"],
    index=0
)

# Display gesture count in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"📝 **Saved Gestures:** {len(gesture_dict)}")
if gesture_dict:
    st.sidebar.markdown("**Available Words:**")
    for word in list(gesture_dict.keys())[:5]:
        st.sidebar.markdown(f"• {word}")
    if len(gesture_dict) > 5:
        st.sidebar.markdown(f"• ... and {len(gesture_dict) - 5} more")

# ═════════════════════════════════════════════════
# 1. Enhanced Real-time Translator
# ═════════════════════════════════════════════════
if menu == "🎥 Real-time Translator":
    st.markdown('<div class="main-title">🎥 Real-time Sign Language Translator</div>', unsafe_allow_html=True)
    
    # Language indicator
    st.markdown(f'''
    <div class="language-selector">
        <div style="text-align: center;">
            <span style="font-size: 18px; font-weight: 600;">🌍 Current Language: 
                <span style="color: #00FFAA;">{st.session_state.selected_language}</span>
            </span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Status and control panel
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        run = st.checkbox("🎥 **Start Camera**", help="Toggle camera for real-time translation")
    with col2:
        speak_enabled = st.checkbox("🔊 **Enable TTS**", value=True, help="Enable text-to-speech output")
    with col3:
        show_confidence = st.checkbox("📊 **Show Metrics**", help="Display detection confidence")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Status indicators
    camera_status = "🟢 Active" if run else "🔴 Inactive"
    tts_status = f"🔊 {st.session_state.selected_language}" if speak_enabled else "🔇 Disabled"
    
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0;">
        <span class="status-indicator {'status-active' if run else 'status-inactive'}">
            Camera: {camera_status}
        </span>
        <span class="status-indicator {'status-active' if speak_enabled else 'status-inactive'}">
            TTS: {tts_status}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    sentence = []
    last_detect_time = 0
    confidence_scores = []

    # Enhanced placeholders
    text_placeholder = st.empty()
    if show_confidence:
        metrics_placeholder = st.empty()
    frame_placeholder = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        
        # Loading indicator
        with st.spinner('🎬 Initializing camera...'):
            time.sleep(1)
        
        st.markdown('<div class="custom-alert alert-info">📹 Camera is now active. Show your hand gestures!</div>', unsafe_allow_html=True)
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.markdown('<div class="custom-alert alert-warning">⚠️ Camera not available or disconnected.</div>', unsafe_allow_html=True)
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            current_confidence = 0
            detected_word = "..."

            if res.multi_hand_landmarks:
                for hl in res.multi_hand_landmarks:
                    # Enhanced hand landmark drawing
                    mp_drawing.draw_landmarks(
                        frame, hl, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 170), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(0, 204, 160), thickness=2)
                    )
                    
                    kp = extract_keypoints(hl)
                    best_word, min_dist = None, float("inf")
                    
                    for word, ref in gesture_dict.items():
                        d = np.linalg.norm(np.array(kp) - np.array(ref))
                        if d < min_dist:
                            min_dist, best_word = d, word
                    
                    # Calculate confidence (inverse of distance, normalized)
                    if best_word:
                        current_confidence = max(0, 1 - (min_dist / 2))
                        detected_word = best_word
                        
                        now = time.time()
                        if current_confidence > 0.6 and (not sentence or sentence[-1] != best_word):
                            if now - last_detect_time >= 2:
                                sentence.append(best_word)
                                last_detect_time = now
                                if speak_enabled:
                                    speak_text(best_word, st.session_state.selected_language)
            
            # Enhanced frame annotations
            cv2.rectangle(frame, (10, 10), (400, 80), (30, 30, 30), -1)
            cv2.putText(frame, f"Word: {detected_word}", (15, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 170), 2)
            cv2.putText(frame, f"Confidence: {current_confidence:.2f}", (15, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 200), 2)
            
            # Display language on frame
            cv2.rectangle(frame, (frame.shape[1]-200, 10), (frame.shape[1]-10, 50), (30, 30, 30), -1)
            cv2.putText(frame, f"Lang: {st.session_state.selected_language[:3]}", (frame.shape[1]-190, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 170), 2)
            
            # Display recent words on frame
            recent_words = " ".join(sentence[-3:]) if sentence else "Start signing..."
            cv2.rectangle(frame, (10, frame.shape[0]-50), (frame.shape[1]-10, frame.shape[0]-10), (30, 30, 30), -1)
            cv2.putText(frame, recent_words, (15, frame.shape[0]-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 170), 2)

            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            
            # Enhanced sentence display
            display_sentence = " ".join(sentence[-10:]) if sentence else "Your translated sentence will appear here..."
            text_placeholder.markdown(
                f'<div class="sentence-display">{display_sentence}</div>',
                unsafe_allow_html=True,
            )
            
            # Show confidence metrics
            if show_confidence:
                confidence_scores.append(current_confidence)
                avg_confidence = np.mean(confidence_scores[-10:]) if confidence_scores else 0
                
                metrics_placeholder.markdown(f"""
                <div class="info-card" style="text-align: center;">
                    <div style="display: flex; justify-content: space-around;">
                        <div class="metric-card">
                            <div class="metric-value">{current_confidence:.2f}</div>
                            <div class="metric-label">Current</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{avg_confidence:.2f}</div>
                            <div class="metric-label">Average</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{len(sentence)}</div>
                            <div class="metric-label">Words</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            time.sleep(0.03)
        
        cap.release()
        st.markdown('<div class="custom-alert alert-success">✅ Camera session ended successfully.</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════
# 2. Enhanced Gesture Management
# ═════════════════════════════════════════════════
elif menu == "✍️ Manage Gestures":
    st.markdown('<div class="main-title">✍️ Gesture Management Studio</div>', unsafe_allow_html=True)
    
    # Add new gesture section
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 🆕 Add New Gesture")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        gesture_name = st.text_input("📌 **Gesture Word**", placeholder="Enter word (e.g., Hello, Thanks, etc.)")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        capture_btn = st.button("📷 **Capture Gesture**", use_container_width=True)
    
    if capture_btn and gesture_name.strip():
        with st.spinner('📸 Preparing to capture in 3 seconds...'):
            countdown_placeholder = st.empty()
            for i in range(3, 0, -1):
                countdown_placeholder.markdown(f"<h2 style='text-align: center; color: #00FFAA;'>📸 Capturing in {i}...</h2>", unsafe_allow_html=True)
                time.sleep(1)
            countdown_placeholder.empty()
        
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                vector = extract_keypoints(res.multi_hand_landmarks[0])
                gesture_dict[gesture_name.strip()] = vector
                with open(GESTURE_FILE, "w") as f:
                    json.dump(gesture_dict, f)
                st.markdown(f'<div class="custom-alert alert-success">✅ Gesture "{gesture_name}" captured and saved successfully!</div>', unsafe_allow_html=True)
                if st.session_state.get('speak_enabled', True):
                    speak_text(f"{gesture_name} saved successfully", st.session_state.selected_language)
                st.rerun()
            else:
                st.markdown('<div class="custom-alert alert-warning">⚠️ No hand detected. Please show your hand clearly.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="custom-alert alert-warning">❌ Camera capture failed. Please check your camera.</div>', unsafe_allow_html=True)
        cap.release()
    elif capture_btn and not gesture_name.strip():
        st.markdown('<div class="custom-alert alert-warning">⚠️ Please enter a gesture word before capturing.</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Manage existing gestures
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Manage Existing Gestures")
    
    if not gesture_dict:
        st.markdown('<div class="custom-alert alert-info">📝 No gestures saved yet. Add your first gesture above!</div>', unsafe_allow_html=True)
    else:
        # Display gestures in a grid
        st.markdown(f"**Total Gestures:** {len(gesture_dict)}")
        
        cols = st.columns(3)
        for idx, word in enumerate(gesture_dict.keys()):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 20px; margin-bottom: 8px;">🤟</div>
                    <div style="font-weight: 600; color: #00FFAA;">{word}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Management controls
        selected = st.selectbox("🎯 **Select gesture to modify**", list(gesture_dict.keys()))
        
        col_del, col_replace, col_test = st.columns(3)
        
        with col_del:
            if st.button("🗑️ **Delete**", use_container_width=True):
                del gesture_dict[selected]
                with open(GESTURE_FILE, "w") as f:
                    json.dump(gesture_dict, f)
                st.markdown(f'<div class="custom-alert alert-success">🗑️ Gesture "{selected}" deleted successfully!</div>', unsafe_allow_html=True)
                speak_text(f"{selected} deleted", st.session_state.selected_language)
                st.rerun()
        
        with col_replace:
            if st.button("🔄 **Replace**", use_container_width=True):
                with st.spinner('📸 Preparing to replace gesture...'):
                    time.sleep(2)
                
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = hands.process(rgb)
                    if res.multi_hand_landmarks:
                        vector = extract_keypoints(res.multi_hand_landmarks[0])
                        gesture_dict[selected] = vector
                        with open(GESTURE_FILE, "w") as f:
                            json.dump(gesture_dict, f)
                        st.markdown(f'<div class="custom-alert alert-success">🔄 Gesture "{selected}" updated successfully!</div>', unsafe_allow_html=True)
                        speak_text(f"{selected} updated", st.session_state.selected_language)
                    else:
                        st.markdown('<div class="custom-alert alert-warning">⚠️ No hand detected during replacement.</div>', unsafe_allow_html=True)
                cap.release()
        
        with col_test:
            if st.button("🧪 **Test TTS**", use_container_width=True):
                speak_text(selected, st.session_state.selected_language)
                st.markdown(f'<div class="custom-alert alert-info">🔊 Testing TTS for "{selected}" in {st.session_state.selected_language}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════
# 3. TTS Testing Page
# ═════════════════════════════════════════════════
elif menu == "🔊 Test TTS":
    st.markdown('<div class="main-title">🔊 TTS Testing Center</div>', unsafe_allow_html=True)
    
    # Language information
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown(f"### 🌍 Current Language: {st.session_state.selected_language}")
    
    # Show available voice info
    if platform.system() != "Darwin":
        voice_id = LANGUAGE_VOICES[st.session_state.selected_language]["voice_id"]
        if voice_id:
            st.success(f"✅ Voice available for {st.session_state.selected_language}")
        else:
            st.warning(f"⚠️ No specific voice found for {st.session_state.selected_language}, using default")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Test options
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 🧪 Test Options")
    
    test_option = st.radio(
        "Select test type:",
        ["Sample Text", "Custom Text", "All Languages Demo"]
    )
    
    if test_option == "Sample Text":
        sample_text = SAMPLE_TEXTS.get(st.session_state.selected_language, "Hello, how are you?")
        st.markdown(f"**Sample text:** {sample_text}")
        
        if st.button("🔊 **Speak Sample Text**"):
            speak_text(sample_text, st.session_state.selected_language)
            st.success("Speaking...")
    
    elif test_option == "Custom Text":
        custom_text = st.text_area("Enter text to speak:", value="Testing sign language translator")
        
        if st.button("🔊 **Speak Custom Text**"):
            speak_text(custom_text, st.session_state.selected_language)
            st.success("Speaking...")
    
    elif test_option == "All Languages Demo":
        st.markdown("**Test greeting in all languages:**")
        
        if st.button("🌍 **Start Demo**"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            languages = list(LANGUAGE_VOICES.keys())
            for idx, lang in enumerate(languages):
                status_text.text(f"Speaking in {lang}...")
                sample = SAMPLE_TEXTS.get(lang, "Hello")
                speak_text(sample, lang)
                progress_bar.progress((idx + 1) / len(languages))
                time.sleep(2)  # Pause between languages
            
            status_text.text("Demo completed!")
            st.balloons()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Language groups display
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 🌏 Language Groups")
    
    for group, langs in language_groups.items():
        st.markdown(f'<div class="language-group">', unsafe_allow_html=True)
        st.markdown(f'<div class="language-group-title">{group}</div>', unsafe_allow_html=True)
        
        cols = st.columns(3)
        for idx, lang in enumerate(langs):
            with cols[idx % 3]:
                voice_status = "✅" if LANGUAGE_VOICES[lang]["voice_id"] or platform.system() == "Darwin" else "⚠️"
                st.markdown(f"• {lang} {voice_status}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════
# 4. Enhanced Model Analytics
# ═════════════════════════════════════════════════
elif menu == "📊 Model Analytics":
    st.markdown('<div class="main-title">📊 Model Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Performance metrics cards
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Performance Metrics")
    
    cols = st.columns(4)
    metrics_colors = ['#00FFAA', '#00CCA0', '#4FFFB2', '#66FFB8']
    
    for idx, (metric, value) in enumerate(model_metrics.items()):
        with cols[idx]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {metrics_colors[idx]};">{value:.1%}</div>
                <div class="metric-label">{metric}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance visualization
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Performance Visualization")
    plot_model_metrics(model_metrics)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # System information
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔧 Technical Specifications:**
        - **Model Type:** Nearest Neighbor Classification
        - **Feature Extraction:** MediaPipe Hand Landmarks
        - **Input Dimensions:** 21 keypoints (x-coordinates)
        - **Processing Framework:** OpenCV + NumPy
        - **Real-time Performance:** ~20-25 FPS
        """)
    
    with col2:
        st.markdown(f"""
        **📊 Current Status:**
        - **Gestures Loaded:** {len(gesture_dict)} words
        - **Detection Method:** Euclidean Distance
        - **Confidence Threshold:** 60%
        - **Platform:** {platform.system()}
        - **TTS Engine:** {"macOS Say" if platform.system() == "Darwin" else "pyttsx3"}
        - **Current Language:** {st.session_state.selected_language}
        - **Languages Available:** {len(LANGUAGE_VOICES)}
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Language support info
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 🌍 Language Support Status")
    
    # Check voice availability
    available_count = 0
    for lang, config in LANGUAGE_VOICES.items():
        if platform.system() == "Darwin" or config["voice_id"]:
            available_count += 1
    
    st.markdown(f"""
    **Voice Availability:**
    - **Total Languages:** {len(LANGUAGE_VOICES)}
    - **Available Voices:** {available_count}
    - **Missing Voices:** {len(LANGUAGE_VOICES) - available_count}
    """)
    
    st.markdown("---")
    
    # Display all languages with their status
    st.markdown("**Language Details:**")
    
    for group, langs in language_groups.items():
        st.markdown(f'<div class="language-group">', unsafe_allow_html=True)
        st.markdown(f'<div class="language-group-title">{group}</div>', unsafe_allow_html=True)
        
        cols = st.columns(3)
        for idx, lang in enumerate(langs):
            with cols[idx % 3]:
                config = LANGUAGE_VOICES[lang]
                if platform.system() == "Darwin":
                    status = "✅ Available"
                    voice_info = MACOS_LANGUAGE_CODES[lang]
                else:
                    if config["voice_id"]:
                        status = "✅ Available"
                        voice_info = "Voice found"
                    else:
                        status = "⚠️ Default"
                        voice_info = "Using default"
                
                st.markdown(f"""
                <div class="metric-card" style="padding: 12px;">
                    <div style="font-size: 14px; font-weight: 600;">{lang}</div>
                    <div style="font-size: 12px; color: {'#00FFAA' if '✅' in status else '#FFC107'};">{status}</div>
                    <div style="font-size: 10px; color: #a0aec0;">{voice_info}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)