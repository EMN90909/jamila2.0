#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  jamila_local.py — Jamila Voice-First AI for Linux          ║
║  Designed for blind and visually impaired users             ║
║  Local LLM support: Gemini, OpenRouter, DeepSeek            ║
║  Persistent chat history with SQLite                        ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import threading
import time
import sqlite3
import subprocess
import re
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

# Load environment variables from .env file
from dotenv import load_dotenv

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
ENV_PATH = SCRIPT_DIR / ".env"

# Load .env file if it exists, otherwise expect system env vars
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=False)
    print(f"✓ Loaded configuration from {ENV_PATH}")
else:
    print("⚠ No .env file found. Using system environment variables.")
    print(f"  Create {ENV_PATH} from .env.example to configure LLM APIs")

# Paths
INSTALL_DIR = Path.home() / ".jamila"
DB_PATH = INSTALL_DIR / "jamila.db"
ICON_PATH = INSTALL_DIR / "jamila.png"

# LLM Configuration from environment
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

# API Keys (loaded from environment)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# OpenRouter specific
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "https://jamila.local")
OPENROUTER_X_TITLE = os.getenv("OPENROUTER_X_TITLE", "Jamila Voice AI")

# Feature flags
ENABLE_CHAT_HISTORY = os.getenv("ENABLE_CHAT_HISTORY", "true").lower() == "true"
MAX_CHAT_HISTORY = int(os.getenv("MAX_CHAT_HISTORY", "50"))

# ─── GLOBALS ─────────────────────────────────────────────────────────────────

_gui_window = None
_speaking_lock = threading.Lock()
_current_chat_id: Optional[str] = None

# ─── DATABASE: PERSISTENT CHAT STORAGE ────────────────────────────────────────

class ChatDatabase:
    """Manages persistent chat history with SQLite."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database with chat history schema."""
        INSTALL_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        
        # Chats table (conversations)
        c.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                chat_id TEXT PRIMARY KEY,
                title TEXT,
                provider TEXT,
                model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Messages table (individual messages)
        c.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT CHECK(role IN ('user', 'assistant', 'system')) NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
            )
        ''')
        
        # Reminders table
        c.execute('''
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                remind_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed BOOLEAN DEFAULT 0
            )
        ''')
        
        # Notes table
        c.execute('''
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for performance
        c.execute('CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_messages_time ON messages(timestamp)')
        
        conn.commit()
        conn.close()
        print("✓ Database initialized")
    
    def create_chat(self, title: str = "New Conversation") -> str:
        """Create a new chat session."""
        chat_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute(
            "INSERT INTO chats (chat_id, title, provider, model) VALUES (?, ?, ?, ?)",
            (chat_id, title, LLM_PROVIDER, LLM_MODEL)
        )
        conn.commit()
        conn.close()
        return chat_id
    
    def get_active_chat(self) -> Optional[str]:
        """Get most recent active chat ID."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute(
            "SELECT chat_id FROM chats WHERE is_active = 1 ORDER BY updated_at DESC LIMIT 1"
        )
        row = c.fetchone()
        conn.close()
        return row[0] if row else None
    
    def add_message(self, chat_id: str, role: str, content: str):
        """Add a message to chat history."""
        if not ENABLE_CHAT_HISTORY:
            return
            
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        
        # Insert message
        c.execute(
            "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
            (chat_id, role, content)
        )
        
        # Update chat timestamp
        c.execute(
            "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE chat_id = ?",
            (chat_id,)
        )
        
        conn.commit()
        conn.close()
    
    def get_chat_history(self, chat_id: str, limit: int = 20) -> List[Dict]:
        """Get recent messages from a chat."""
        if not ENABLE_CHAT_HISTORY:
            return []
            
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute(
            """SELECT role, content, timestamp FROM messages 
               WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?""",
            (chat_id, limit)
        )
        
        rows = c.fetchall()
        conn.close()
        
        # Return in chronological order
        return [dict(row) for row in reversed(rows)]
    
    def get_all_chats(self) -> List[Dict]:
        """Get list of all chats."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute(
            """SELECT chat_id, title, created_at, updated_at, 
                      (SELECT COUNT(*) FROM messages WHERE messages.chat_id = chats.chat_id) as message_count
               FROM chats ORDER BY updated_at DESC"""
        )
        
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def delete_chat(self, chat_id: str):
        """Delete a chat and all its messages."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
        conn.commit()
        conn.close()
    
    def update_chat_title(self, chat_id: str, title: str):
        """Update chat title."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("UPDATE chats SET title = ? WHERE chat_id = ?", (title, chat_id))
        conn.commit()
        conn.close()
    
    # Reminders
    def add_reminder(self, text: str, remind_at: Optional[str] = None):
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("INSERT INTO reminders (text, remind_at) VALUES (?, ?)", (text, remind_at))
        conn.commit()
        conn.close()
    
    def get_reminders(self) -> List[Tuple]:
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("SELECT text, remind_at FROM reminders WHERE completed = 0 ORDER BY remind_at")
        rows = c.fetchall()
        conn.close()
        return rows
    
    # Notes
    def add_note(self, title: str, content: str):
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("INSERT INTO notes (title, content) VALUES (?, ?)", (title, content))
        conn.commit()
        conn.close()
    
    def get_notes(self) -> List[Tuple]:
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("SELECT title, content FROM notes ORDER BY updated_at DESC")
        rows = c.fetchall()
        conn.close()
        return rows

# Initialize database
db = ChatDatabase(DB_PATH)

# ─── TTS SETUP ────────────────────────────────────────────────────────────────

_tts_engine = None
_tts_mode = None

def init_tts():
    global _tts_engine, _tts_mode
    
    # Try Coqui TTS first (neural, natural)
    try:
        from TTS.api import TTS as CoquiTTS
        print("→ Loading Coqui TTS neural voice model...")
        _tts_engine = CoquiTTS(
            model_name="tts_models/en/ljspeech/tacotron2-DDC", 
            progress_bar=False, 
            gpu=False
        )
        _tts_mode = "coqui"
        print("✓ Coqui TTS ready")
        return
    except Exception as e:
        print(f"  Coqui TTS not available ({e}), trying espeak...")

    # Try espeak-ng
    try:  
        result = subprocess.run(["espeak-ng", "--version"], capture_output=True, timeout=3)  
        if result.returncode == 0:  
            _tts_mode = "espeak"  
            print("✓ espeak-ng TTS ready")  
            return  
    except Exception: 
        pass  

    # Fall back to pyttsx3  
    try:  
        import pyttsx3  
        _tts_engine = pyttsx3.init()  
        _tts_engine.setProperty('rate', 155)  
        _tts_engine.setProperty('volume', 0.95)  
        _tts_mode = "pyttsx3"  
        print("✓ pyttsx3 TTS ready")  
        return  
    except Exception as e:  
        print(f"⚠ No TTS available: {e}")  
        _tts_mode = "print"

def speak(text: str, blocking: bool = True):
    """Speak text aloud using the best available TTS engine."""
    if not text or not text.strip():
        return

    if _gui_window:  
        _gui_window.set_response(text)  

    clean = text.strip()  
    print(f"\n🔊 Jamila: {clean}\n")  

    with _speaking_lock:
        if _tts_mode == "coqui" and _tts_engine:  
            try:  
                import tempfile, soundfile as sf, sounddevice as sd  
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:  
                    tmp = f.name  
                _tts_engine.tts_to_file(text=clean, file_path=tmp)  
                data, samplerate = sf.read(tmp)  
                sd.play(data, samplerate)  
                if blocking: 
                    sd.wait()  
                os.unlink(tmp)  
                return  
            except Exception as e:  
                print(f"  Coqui speak error: {e}")  

        if _tts_mode == "espeak":  
            try:  
                cmd = ["espeak-ng", "-s", "145", "-p", "50", "-v", "en", clean]  
                if blocking:  
                    subprocess.run(cmd, capture_output=True)  
                else:  
                    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  
                return  
            except Exception as e:  
                print(f"  espeak error: {e}")  

        if _tts_mode == "pyttsx3" and _tts_engine:  
            try:  
                _tts_engine.say(clean)  
                _tts_engine.runAndWait()  
                return  
            except Exception: 
                pass

# ─── SPEECH RECOGNITION ───────────────────────────────────────────────────────

_recognizer = None
_microphone = None
_listening = False

def init_stt():
    global _recognizer, _microphone
    try:
        import speech_recognition as sr
        _recognizer = sr.Recognizer()
        _recognizer.dynamic_energy_threshold = True
        _recognizer.pause_threshold = 0.8
        _microphone = sr.Microphone()
        with _microphone as source:
            _recognizer.adjust_for_ambient_noise(source, duration=1)
        print("✓ Microphone ready")
        return True
    except Exception as e:
        print(f"⚠ Microphone not available: {e}")
        return False

def listen_once(timeout: int = 8) -> Optional[str]:
    """Listen for one spoken command. Returns text or None."""
    global _listening
    if not _recognizer or not _microphone:
        return None
    
    import speech_recognition as sr
    _listening = True
    
    if _gui_window: 
        _gui_window.set_listening(True)
    
    try:
        with _microphone as source:
            audio = _recognizer.listen(source, timeout=timeout, phrase_time_limit=15)
            text = _recognizer.recognize_google(audio)
            return text.strip()
    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        return None
    except Exception as e:
        print(f"STT error: {e}")
        return None
    finally:
        _listening = False
        if _gui_window: 
            _gui_window.set_listening(False)

# ─── LLM CLIENTS ─────────────────────────────────────────────────────────────

class LLMClient:
    """Unified interface for multiple LLM providers."""
    
    @staticmethod
    def chat(messages: List[Dict[str, str]], model: Optional[str] = None) -> Tuple[str, bool]:
        """
        Send chat completion request to configured LLM.
        Returns: (response_text, success)
        """
        provider = LLM_PROVIDER
        model = model or LLM_MODEL
        
        try:
            if provider == "gemini":
                return LLMClient._chat_gemini(messages, model)
            elif provider == "openrouter":
                return LLMClient._chat_openrouter(messages, model)
            elif provider == "deepseek":
                return LLMClient._chat_deepseek(messages, model)
            else:
                return f"Unknown provider: {provider}", False
        except Exception as e:
            return f"Error communicating with {provider}: {str(e)}", False
    
    @staticmethod
    def _chat_gemini(messages: List[Dict[str, str]], model: str) -> Tuple[str, bool]:
        """Google Gemini API client."""
        if not GEMINI_API_KEY:
            return "Gemini API key not configured. Set GEMINI_API_KEY in .env", False
        
        # Convert messages to Gemini format
        # Gemini uses 'user' and 'model' roles
        gemini_messages = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        # Use the latest Gemini API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}
        
        payload = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048
            }
        }
        
        response = requests.post(url, headers=headers, params=params, json=payload, timeout=60)
        data = response.json()
        
        if response.status_code != 200:
            error_msg = data.get("error", {}).get("message", "Unknown error")
            return f"Gemini API error: {error_msg}", False
        
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return text, True
        except (KeyError, IndexError) as e:
            return f"Error parsing Gemini response: {str(e)}", False
    
    @staticmethod
    def _chat_openrouter(messages: List[Dict[str, str]], model: str) -> Tuple[str, bool]:
        """OpenRouter API client (OpenAI compatible)."""
        if not OPENROUTER_API_KEY:
            return "OpenRouter API key not configured. Set OPENROUTER_API_KEY in .env", False
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_HTTP_REFERER,
            "X-Title": OPENROUTER_X_TITLE
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        data = response.json()
        
        if response.status_code != 200:
            error_msg = data.get("error", {}).get("message", "Unknown error")
            return f"OpenRouter API error: {error_msg}", False
        
        try:
            text = data["choices"][0]["message"]["content"]
            return text, True
        except (KeyError, IndexError) as e:
            return f"Error parsing OpenRouter response: {str(e)}", False
    
    @staticmethod
    def _chat_deepseek(messages: List[Dict[str, str]], model: str) -> Tuple[str, bool]:
        """DeepSeek API client (OpenAI compatible)."""
        if not DEEPSEEK_API_KEY:
            return "DeepSeek API key not configured. Set DEEPSEEK_API_KEY in .env", False
        
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model or "deepseek-chat",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        data = response.json()
        
        if response.status_code != 200:
            error_msg = data.get("error", {}).get("message", "Unknown error")
            return f"DeepSeek API error: {error_msg}", False
        
        try:
            text = data["choices"][0]["message"]["content"]
            return text, True
        except (KeyError, IndexError) as e:
            return f"Error parsing DeepSeek response: {str(e)}", False

# ─── GUI WINDOW ───────────────────────────────────────────────────────────────

class JamilaWindow:
    """GTK main window — accessible, voice-centric UI."""

    def __init__(self):  
        import gi  
        gi.require_version('Gtk', '3.0')  
        from gi.repository import Gtk, GLib, GdkPixbuf, Pango, Gdk, Atk
        
        self.Gtk = Gtk  
        self.GLib = GLib  
        self.Pango = Pango  
        self.Gdk = Gdk

        self.window = Gtk.Window()  
        self.window.set_title("Jamila — Local Voice AI")  
        self.window.set_default_size(600, 700)  
        self.window.set_border_width(0)  
        self.window.connect("destroy", Gtk.main_quit)  
        self.window.set_resizable(True)  
        
        # Accessibility
        self.window.get_accessible().set_role(Atk.Role.APPLICATION)
        self.window.get_accessible().set_description("Jamila Local Voice AI Assistant")

        # CSS styling
        css = b"""  
        window { background-color: #0a0a08; }  
        .header-box { background-color: #111108; border-bottom: 1px solid #2a2a1e; }  
        .response-area { background-color: #0f0f0a; border: 1px solid #2a2a1e; border-radius: 12px; padding: 16px; }  
        .status-label { font-size: 13px; }  
        .mic-button {  
            background: #c8973a;  
            border-radius: 50px;  
            border: none;  
            color: #0a0a08;  
            font-size: 22px;  
            font-weight: bold;  
            padding: 20px 40px;  
            transition: all 0.2s;  
        }  
        .mic-button:hover { background: #e8b84b; }  
        .mic-button.listening { background: #e74c3c; }  
        .type-button { background: #1e1e18; border: 1px solid #3a3a2e; border-radius: 8px; color: #c8c8a0; padding: 10px 20px; }  
        .type-button:hover { background: #2a2a20; }  
        .cmd-entry {  
            background-color: #1a1a12;  
            border: 1px solid #3a3a2e;  
            border-radius: 8px;  
            color: #f5f0e8;  
            font-family: monospace;  
            font-size: 14px;  
            padding: 10px 14px;  
        }  
        .response-label { font-size: 16px; line-height: 1.6; color: #f5f0e8; }  
        .history-label { font-size: 13px; color: #888870; }  
        .provider-label { font-size: 11px; color: #666650; }
        """  
        css_provider = Gtk.CssProvider()  
        css_provider.load_from_data(css)  
        Gtk.StyleContext.add_provider_for_screen(  
            Gdk.Screen.get_default(),  
            css_provider,  
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION  
        )  

        # Main layout
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)  
        self.window.add(main_box)  

        # ── HEADER ──  
        header = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)  
        header.get_style_context().add_class('header-box')  
        header.set_border_width(20)  
        main_box.pack_start(header, False, False, 0)  

        # Logo row
        logo_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)  
        header.pack_start(logo_row, False, False, 0)  

        # Icon
        try:  
            icon_buf = GdkPixbuf.Pixbuf.new_from_file_at_scale(str(ICON_PATH), 52, 52, True)  
            icon_img = Gtk.Image.new_from_pixbuf(icon_buf)  
            logo_row.pack_start(icon_img, False, False, 0)  
        except Exception:  
            icon_lbl = Gtk.Label("🎙")  
            icon_lbl.set_markup('<span font="32">🎙</span>')  
            logo_row.pack_start(icon_lbl, False, False, 0)  

        name_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)  
        logo_row.pack_start(name_box, False, False, 0)  

        title_lbl = Gtk.Label()  
        title_lbl.set_markup('<span font="Fraunces, serif" size="20000" weight="heavy" foreground="#f5f0e8">Ja<i><span foreground="#c8973a">mila</span></i></span>')  
        title_lbl.set_xalign(0)  
        name_box.pack_start(title_lbl, False, False, 0)  

        tagline_lbl = Gtk.Label()  
        tagline_lbl.set_markup('<span size="10000" foreground="#888870">Local Voice AI · Private & Persistent</span>')  
        tagline_lbl.set_xalign(0)  
        name_box.pack_start(tagline_lbl, False, False, 0)  

        # Status
        self.status_label = Gtk.Label("Ready")  
        self.status_label.get_style_context().add_class('status-label')  
        self.status_label.set_markup('<span foreground="#27ae60" size="10000">● Ready</span>')  
        logo_row.pack_end(self.status_label, False, False, 0)  

        # Provider info
        self.provider_label = Gtk.Label()
        self.provider_label.set_markup(
            f'<span foreground="#666650" size="9000">Provider: {LLM_PROVIDER.upper()} | Model: {LLM_MODEL}</span>'
        )
        self.provider_label.set_xalign(0)
        header.pack_start(self.provider_label, False, False, 0)

        # ── RESPONSE AREA ──  
        resp_wrapper = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)  
        resp_wrapper.set_border_width(16)  
        main_box.pack_start(resp_wrapper, True, True, 0)  

        resp_title = Gtk.Label()  
        resp_title.set_markup('<span foreground="#c8973a" size="9000" weight="bold" letter_spacing="2000">JAMILA\'S RESPONSE</span>')  
        resp_title.set_xalign(0)  
        resp_wrapper.pack_start(resp_title, False, False, 0)  

        scrolled = Gtk.ScrolledWindow()  
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)  
        scrolled.set_vexpand(True)  
        resp_wrapper.pack_start(scrolled, True, True, 0)  

        resp_viewport = Gtk.Viewport()  
        resp_viewport.get_style_context().add_class('response-area')  
        scrolled.add(resp_viewport)  

        self.response_label = Gtk.Label()  
        self.response_label.set_markup(
            '<span foreground="#555540" size="14000" style="italic">Welcome to Jamila! Press the microphone button or type a command.\n\n'
            'Try: "What time is it?", "Add reminder call mom", or "Tell me a joke"</span>'
        )  
        self.response_label.set_line_wrap(True)  
        self.response_label.set_line_wrap_mode(Pango.WrapMode.WORD_CHAR)  
        self.response_label.set_xalign(0)  
        self.response_label.set_yalign(0)  
        self.response_label.set_selectable(True)
        self.response_label.set_margin_start(4)  
        self.response_label.set_margin_top(4)  
        self.response_label.get_style_context().add_class('response-label')
        self.response_label.get_accessible().set_role(Atk.Role.LABEL)
        resp_viewport.add(self.response_label)  

        # Chat history area
        hist_title = Gtk.Label()  
        hist_title.set_markup('<span foreground="#c8973a" size="9000" weight="bold" letter_spacing="2000">CONVERSATION HISTORY</span>')  
        hist_title.set_xalign(0)  
        resp_wrapper.pack_start(hist_title, False, False, 0)  

        history_scroll = Gtk.ScrolledWindow()  
        history_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)  
        history_scroll.set_size_request(-1, 150)  
        resp_wrapper.pack_start(history_scroll, False, False, 0)  

        self.history_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)  
        history_scroll.add(self.history_box)  

        # ── CONTROLS ──  
        controls = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)  
        controls.set_border_width(16)  
        main_box.pack_start(controls, False, False, 0)  

        # Mic button
        self.mic_btn = Gtk.Button()  
        self.mic_btn.get_style_context().add_class('mic-button')
        self.mic_btn.get_accessible().set_name("Speak button")
        self.mic_btn.get_accessible().set_description("Press to speak a command")
        self.mic_label = Gtk.Label()  
        self.mic_label.set_markup('<span size="18000" weight="bold">🎙  Press to Speak</span>')  
        self.mic_btn.add(self.mic_label)  
        self.mic_btn.connect("clicked", self.on_mic_click)  
        self.mic_btn.set_tooltip_text("Click to speak (or press Space)")  
        controls.pack_start(self.mic_btn, False, False, 0)  

        self.window.connect("key-press-event", self.on_key_press)  
        self.mic_btn.set_can_focus(True)  

        sep = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)  
        controls.pack_start(sep, False, False, 0)  

        # Type command row  
        type_label = Gtk.Label()  
        type_label.set_markup('<span foreground="#888870" size="10000">Or type a command:</span>')  
        type_label.set_xalign(0)  
        controls.pack_start(type_label, False, False, 0)  

        cmd_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)  
        controls.pack_start(cmd_row, False, False, 0)  

        self.cmd_entry = Gtk.Entry()  
        self.cmd_entry.get_style_context().add_class('cmd-entry')
        self.cmd_entry.get_accessible().set_name("Command input")
        self.cmd_entry.get_accessible().set_description("Type a command and press Enter")
        self.cmd_entry.set_placeholder_text("Type here and press Enter...")  
        self.cmd_entry.connect("activate", self.on_cmd_enter)  
        cmd_row.pack_start(self.cmd_entry, True, True, 0)  

        send_btn = Gtk.Button()  
        send_btn.get_style_context().add_class('type-button')  
        send_btn.add(Gtk.Label("Send →"))  
        send_btn.connect("clicked", self.on_cmd_enter)  
        cmd_row.pack_start(send_btn, False, False, 0)  

        # Quick commands  
        quick_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)  
        controls.pack_start(quick_row, False, False, 0)  
        
        quick_cmds = [
            ("Help", "help"), 
            ("New Chat", "new chat"), 
            ("History", "show history"),
            ("Reminders", "list reminders"), 
            ("Notes", "list notes")
        ]
        
        for label, cmd in quick_cmds:  
            btn = Gtk.Button(label=label)  
            btn.get_style_context().add_class('type-button')
            btn.get_accessible().set_name(label + " button")
            btn.connect("clicked", lambda b, c=cmd: self.run_cmd(c))  
            quick_row.pack_start(btn, True, True, 0)  

        self.window.show_all()
        
        # Load recent history if continuing chat
        self._load_recent_history()

    def _load_recent_history(self):
        """Load recent chat history into UI."""
        global _current_chat_id
        if _current_chat_id:
            history = db.get_chat_history(_current_chat_id, limit=10)
            for msg in history:
                self.add_history(msg["role"], msg["content"], speak=False)

    def set_status(self, text: str, color: str = "#c8973a"):  
        self.GLib.idle_add(lambda: self.status_label.set_markup(f'<span foreground="{color}" size="10000">● {text}</span>') or False)  

    def set_listening(self, on: bool):  
        if on:  
            self.GLib.idle_add(lambda: self.mic_label.set_markup('<span size="18000" weight="bold">🔴  Listening...</span>') or False)  
            self.GLib.idle_add(lambda: self.mic_btn.get_style_context().add_class('listening') or False)  
            self.set_status("Listening...", "#e74c3c")  
        else:  
            self.GLib.idle_add(lambda: self.mic_label.set_markup('<span size="18000" weight="bold">🎙  Press to Speak</span>') or False)  
            self.GLib.idle_add(lambda: self.mic_btn.get_style_context().remove_class('listening') or False)  
            self.set_status("Ready", "#27ae60")  

    def set_response(self, text: str):  
        def _update():  
            escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')  
            self.response_label.set_markup(f'<span foreground="#f5f0e8" size="15000">{escaped}</span>')
            self.response_label.get_accessible().set_description(escaped)
            return False  
        self.GLib.idle_add(_update)  

    def set_thinking(self):  
        self.set_response("Thinking...")  
        self.set_status("Thinking...", "#c8973a")  

    def add_history(self, role: str, text: str, speak: bool = True):  
        def _update():  
            color = "#c8973a" if role == "user" else "#888870"  
            prefix = "You:" if role == "user" else "Jamila:"
            short = text[:120] + ("..." if len(text) > 120 else "")  
            escaped = short.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')  
            lbl = self.Gtk.Label()  
            lbl.set_markup(f'<span foreground="{color}" size="10000" weight="bold">{prefix}</span> <span foreground="#aaa890" size="10000">{escaped}</span>')  
            lbl.set_xalign(0)  
            lbl.set_line_wrap(True)  
            self.history_box.pack_start(lbl, False, False, 0)  
            self.history_box.show_all()
            return False  
        self.GLib.idle_add(_update)
        
        # Speak the assistant response if requested
        if speak and role == "assistant":
            speak(text)

    def on_mic_click(self, btn):  
        threading.Thread(target=self._do_listen, daemon=True).start()  

    def on_key_press(self, widget, event):  
        import gi
        from gi.repository import Gdk  
        if event.keyval == Gdk.KEY_space and not self.cmd_entry.has_focus():  
            self.on_mic_click(None)  
            return True  
        return False  

    def on_cmd_enter(self, widget):  
        cmd = self.cmd_entry.get_text().strip()  
        if cmd:  
            self.cmd_entry.set_text("")  
            threading.Thread(target=self.run_cmd, args=(cmd,), daemon=True).start()  

    def _do_listen(self):  
        speak("Listening", blocking=False)  
        text = listen_once()  
        if text:  
            self.add_history("user", text, speak=False)
            self.run_cmd(text)  
        else:  
            speak("I didn't hear anything. Please try again.")  

    def run_cmd(self, cmd: str):  
        result = parse_and_run(cmd)  
        if result == 'exit':  
            self.GLib.idle_add(self.Gtk.main_quit)
        elif result and isinstance(result, str):
            self.add_history("assistant", result)

def run_gui():
    global _gui_window
    try:
        import gi
        gi.require_version('Gtk', '3.0')
        _gui_window = JamilaWindow()
        _gui_window.set_status("Ready", "#27ae60")
        from gi.repository import Gtk
        Gtk.main()
        return True
    except Exception as e:
        print(f"GUI not available ({e}), falling back to terminal mode")
        return False

# ─── COMMAND PROCESSOR ─────────────────────────────────────────────────────────

def parse_and_run(cmd: str) -> Optional[str]:
    """
    Parse and execute user commands.
    Returns: 'exit', response text, or None
    """
    global _current_chat_id
    
    cmd_lower = cmd.lower().strip()
    words = cmd_lower.split()
    
    # Exit commands
    if any(x in cmd_lower for x in ['exit', 'quit', 'goodbye', 'bye', 'shutdown']):
        speak("Goodbye! Jamila is shutting down.")
        return 'exit'
    
    # Help
    if cmd_lower in ['help', 'what can you do', 'commands', 'assist']:
        help_text = (
            "I can help you with: general conversation, reminders, notes, time and date, "
            "and managing chat history. "
            "Local commands: 'new chat' to start fresh, 'show history' for past chats, "
            "'add reminder', 'list reminders', 'add note', 'list notes'. "
            "I remember our conversation context within each chat session."
        )
        speak(help_text)
        return help_text
    
    # New chat
    if cmd_lower in ['new chat', 'start new chat', 'clear chat', 'reset']:
        _current_chat_id = db.create_chat(title=cmd[:50])
        speak("Started a new conversation. How can I help you?")
        return "Started new chat"
    
    # Show chat history list
    if cmd_lower in ['show history', 'chat history', 'past chats', 'previous chats']:
        chats = db.get_all_chats()
        if chats:
            speak(f"You have {len(chats)} saved conversations.")
            for i, chat in enumerate(chats[:3], 1):
                print(f"  {i}. {chat['title']} ({chat['message_count']} messages)")
            return f"You have {len(chats)} saved chats"
        else:
            speak("No saved conversations yet.")
            return "No chat history"
    
    # Time
    if any(x in cmd_lower for x in ['time', 'what time', 'current time']):
        now = datetime.now().strftime("%I:%M %p")
        speak(f"The current time is {now}")
        return f"The current time is {now}"
    
    # Date
    if any(x in cmd_lower for x in ['date', 'what day', 'today', 'current date']):
        today = datetime.now().strftime("%A, %B %d, %Y")
        speak(f"Today is {today}")
        return f"Today is {today}"
    
    # Reminders
    if 'reminder' in cmd_lower:
        if any(x in cmd_lower for x in ['add', 'set', 'create']):
            # Extract reminder text
            text = cmd
            for prefix in ['add reminder', 'set reminder', 'create reminder', 'reminder to', 'reminder']:
                if prefix in cmd_lower:
                    text = cmd[cmd_lower.find(prefix) + len(prefix):].strip()
                    break
            if text:
                db.add_reminder(text)
                speak(f"Reminder added: {text}")
                return f"Reminder added: {text}"
            else:
                speak("What would you like me to remind you about?")
                return "What should I remind you?"
        elif any(x in cmd_lower for x in ['list', 'show', 'what']):
            reminders = db.get_reminders()
            if reminders:
                text = f"You have {len(reminders)} reminders: " + ". ".join([r[0] for r in reminders[:5]])
                speak(text)
                return text
            else:
                speak("You have no reminders.")
                return "No reminders"
    
    # Notes
    if 'note' in cmd_lower:
        if any(x in cmd_lower for x in ['add', 'create', 'take']):
            text = cmd
            for prefix in ['add note', 'create note', 'take note', 'note that', 'note']:
                if prefix in cmd_lower:
                    text = cmd[cmd_lower.find(prefix) + len(prefix):].strip()
                    break
            if text:
                title = text[:30] + "..." if len(text) > 30 else text
                db.add_note(title, text)
                speak(f"Note saved: {text[:50]}")
                return f"Note saved: {text[:50]}"
            else:
                speak("What would you like to note down?")
                return "What note?"
        elif any(x in cmd_lower for x in ['list', 'show', 'what']):
            notes = db.get_notes()
            if notes:
                text = f"You have {len(notes)} notes. Latest: {notes[0][1][:100]}"
                speak(text)
                return text
            else:
                speak("You have no notes.")
                return "No notes"
    
    # Default: AI conversation
    return handle_ai_chat(cmd)

def handle_ai_chat(user_input: str) -> str:
    """Handle AI conversation with context."""
    global _current_chat_id
    
    # Create new chat if none exists
    if not _current_chat_id:
        _current_chat_id = db.create_chat(title=user_input[:50])
    
    # Save user message
    db.add_message(_current_chat_id, "user", user_input)
    
    # Build message history for context
    messages = [{"role": "system", "content": "You are Jamila, a helpful voice-first AI assistant designed for blind and visually impaired users. Keep responses concise, clear, and conversational. Avoid visual references and format for spoken delivery."}]
    
    # Add recent history for context
    history = db.get_chat_history(_current_chat_id, limit=MAX_CHAT_HISTORY)
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current message (already in history but ensure it's last)
    if not history or history[-1]["content"] != user_input:
        messages.append({"role": "user", "content": user_input})
    
    # Get AI response
    if _gui_window:
        _gui_window.set_thinking()
    
    response, success = LLMClient.chat(messages)
    
    if success:
        # Save assistant response
        db.add_message(_current_chat_id, "assistant", response)
        speak(response)
        return response
    else:
        # Error message
        speak(response)  # speak the error
        return response

# ─── MAIN ENTRY POINT ─────────────────────────────────────────────────────────

def check_configuration():
    """Verify that at least one LLM is configured."""
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        return False, "Gemini API key not configured. Set GEMINI_API_KEY in .env file."
    if LLM_PROVIDER == "openrouter" and not OPENROUTER_API_KEY:
        return False, "OpenRouter API key not configured. Set OPENROUTER_API_KEY in .env file."
    if LLM_PROVIDER == "deepseek" and not DEEPSEEK_API_KEY:
        return False, "DeepSeek API key not configured. Set DEEPSEEK_API_KEY in .env file."
    return True, "Configuration valid"

def main():
    """Main entry point for Jamila."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Jamila Voice-First AI Assistant (Local LLM Edition)        ║")
    print("║  Supports: Gemini, OpenRouter, DeepSeek                     ║")
    print("║  Persistent chat history enabled                            ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    
    # Check configuration
    ok, msg = check_configuration()
    if not ok:
        print(f"⚠ Configuration Error: {msg}")
        print(f"\nPlease create {ENV_PATH} from .env.example and add your API keys.")
        print("Continuing in limited mode (local commands only)...\n")
        speak("Configuration incomplete. Please set up your API keys in the environment file.")
    else:
        print(f"✓ Using LLM Provider: {LLM_PROVIDER.upper()}")
        print(f"✓ Model: {LLM_MODEL}")
        print(f"✓ Chat history: {'Enabled' if ENABLE_CHAT_HISTORY else 'Disabled'}")
    
    # Initialize systems
    init_tts()
    mic_ok = init_stt()
    
    # Welcome message
    speak("Welcome to Jamila. I'm ready to help you. Press space or click the microphone button to speak.")
    
    # Try GUI mode
    if not run_gui():
        # Fallback to terminal mode
        print("\n--- Terminal Mode ---")
        print("Commands: 'exit' to quit, 'help' for commands, 'new chat' to reset context")
        
        global _current_chat_id
        _current_chat_id = db.create_chat(title="Terminal Session")
        
        while True:
            try:
                if mic_ok:
                    print("\n[Press Enter to speak, or type a command]")
                    user_input = input("> ").strip()
                    if not user_input:
                        text = listen_once()
                        if text:
                            print(f"You said: {text}")
                            result = parse_and_run(text)
                            if result == 'exit':
                                break
                    else:
                        result = parse_and_run(user_input)
                        if result == 'exit':
                            break
                else:
                    user_input = input("Jamila> ").strip()
                    result = parse_and_run(user_input)
                    if result == 'exit':
                        break
            except KeyboardInterrupt:
                speak("Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
