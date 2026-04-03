import cv2
import os
import re
import tempfile
import json
import wave
import time
import threading
from pathlib import Path
import uuid
from collections import deque
from dotenv import load_dotenv

from flask import Flask, render_template, redirect, request, url_for, flash, session, Response, g, jsonify
from flask_mysqldb import MySQL
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import torch
import numpy as np
from Model.features import FEATURE_SIZE, SEQUENCE_LENGTH, extract_landmark_features, normalize_sequence
from Model.gesture_model import GestureTransformer

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Secret key for session management
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev_secret_key')

# MySQL Configuration from environment variables
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', '')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB', 'sign_language_app')
app.config['MYSQL_PORT'] = int(os.getenv('MYSQL_PORT', '3306'))
app.config['GOOGLE_TRANSLATE_API_KEY'] = os.getenv('GOOGLE_TRANSLATE_API_KEY', '')

mysql = MySQL(app)


@app.context_processor
def inject_frontend_config():
    return {
        "google_translate_api_key": app.config.get("GOOGLE_TRANSLATE_API_KEY", ""),
        "emailjs_service_id": os.getenv("VITE_EMAILJS_SERVICE_ID", ""),
        "emailjs_template_id": os.getenv("VITE_EMAILJS_TEMPLATE_ID", ""),
        "emailjs_public_key": os.getenv("VITE_EMAILJS_PUBLIC_KEY", ""),
    }


# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')


# Registration Form
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[Length(min=4, max=25), DataRequired()])
    email = StringField('Email', validators=[Email(), DataRequired()])
    password = PasswordField('Password', validators=[
        DataRequired(),
        EqualTo('confirm', message='Passwords must match')
    ])
    confirm = PasswordField('Confirm Password', validators=[DataRequired()])
    submit = SubmitField('Register')

# Login Form
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[Email(), DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

ASSET_DIR = Path(app.root_path) / "static" / "assets"
_CONTRACTION_PATTERNS = [
    (re.compile(r"\bwon't\b", re.IGNORECASE), "will not"),
    (re.compile(r"\bcan't\b", re.IGNORECASE), "cannot"),
    (re.compile(r"n't\b", re.IGNORECASE), " not"),
    (re.compile(r"'re\b", re.IGNORECASE), " are"),
    (re.compile(r"'ve\b", re.IGNORECASE), " have"),
    (re.compile(r"'ll\b", re.IGNORECASE), " will"),
    (re.compile(r"'d\b", re.IGNORECASE), " would"),
    (re.compile(r"'m\b", re.IGNORECASE), " am"),
    (re.compile(r"'s\b", re.IGNORECASE), " is"),
]


def _normalize_asset_key(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def _build_asset_maps():
    single_word = {}
    multi_word = {}
    alpha_numeric = set()

    if not ASSET_DIR.exists():
        return single_word, multi_word, alpha_numeric

    for mp4 in ASSET_DIR.glob("*.mp4"):
        stem = mp4.stem
        key = _normalize_asset_key(stem)
        if not key:
            continue

        if " " in key:
            multi_word[key] = stem
        else:
            single_word[key] = stem
            if len(stem) == 1 and stem.isalnum():
                alpha_numeric.add(stem.upper())

    return single_word, multi_word, alpha_numeric


ASSET_SINGLE_MAP, ASSET_MULTI_MAP, ASSET_ALPHA_NUMERIC = _build_asset_maps()
ASSET_MULTI_MAX_WORDS = max((len(k.split()) for k in ASSET_MULTI_MAP), default=1)
DEFAULT_SIGN_STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "for", "of", "to", "at", "in", "on", "by", "with", "from", "as",
    "and", "or", "if", "then", "than",
}
STOPWORDS_FILE = Path(app.root_path) / "config" / "sign_stopwords.txt"


def _load_sign_stopwords():
    if not STOPWORDS_FILE.exists():
        return set(DEFAULT_SIGN_STOPWORDS)

    words = set()
    for raw_line in STOPWORDS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lower()
        if not line or line.startswith("#"):
            continue
        words.add(line)

    return words if words else set(DEFAULT_SIGN_STOPWORDS)


SIGN_STOPWORDS = _load_sign_stopwords()


def _expand_contractions(text):
    normalized = text.lower()
    for pattern, replacement in _CONTRACTION_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def _candidate_tokens(token):
    candidates = [token]
    if token.endswith("ies") and len(token) > 3:
        candidates.append(token[:-3] + "y")
    if token.endswith("ing") and len(token) > 4:
        base = token[:-3]
        candidates.append(base)
        candidates.append(base + "e")
    if token.endswith("ed") and len(token) > 3:
        base = token[:-2]
        candidates.append(base)
        candidates.append(base + "e")
    if token.endswith("es") and len(token) > 3:
        candidates.append(token[:-2])
    if token.endswith("s") and len(token) > 2:
        candidates.append(token[:-1])

    deduped = []
    seen = set()
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _resolve_token_to_asset(token):
    for candidate in _candidate_tokens(token):
        if candidate in ASSET_SINGLE_MAP:
            return ASSET_SINGLE_MAP[candidate]
    return None


def process_text_for_animation(text, remove_stopwords=True):
    if not text:
        return []

    normalized = _expand_contractions(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    if not tokens:
        return []

    final_words = []
    i = 0

    while i < len(tokens):
        matched_phrase = None
        max_window = min(ASSET_MULTI_MAX_WORDS, len(tokens) - i)

        for window in range(max_window, 1, -1):
            phrase_key = " ".join(tokens[i:i + window])
            if phrase_key in ASSET_MULTI_MAP:
                matched_phrase = ASSET_MULTI_MAP[phrase_key]
                i += window
                break

        if matched_phrase:
            final_words.append(matched_phrase)
            continue

        token = tokens[i]

        if remove_stopwords and token in SIGN_STOPWORDS:
            i += 1
            continue

        resolved = _resolve_token_to_asset(token)

        if resolved:
            final_words.append(resolved)
        else:
            for ch in token.upper():
                if ch in ASSET_ALPHA_NUMERIC:
                    final_words.append(ch)

        i += 1

    return final_words


# Routes
@app.route('/')
def home():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    return render_template('home.html', username=None)

@app.route('/index')
def index():
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    flash('Please log in first', 'danger')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = generate_password_hash(form.password.data)

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users(username, email, password) VALUES(%s, %s, %s)", (username, email, password))
        mysql.connection.commit()
        cur.close()

        flash('You are now registered and can log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password_candidate = form.password.data

        cur = mysql.connection.cursor()
        result = cur.execute("SELECT id, username, email, password FROM users WHERE email = %s", [email])

        if result > 0:
            data = cur.fetchone()
            user_id, username, stored_email, stored_password = data

            if check_password_hash(stored_password, password_candidate):
                session['logged_in'] = True
                session['username'] = username
                session['email'] = stored_email
                session['user_id'] = user_id
                
                # Session and Login History
                session_id = str(uuid.uuid4())
                cur.execute("INSERT INTO sessions (session_id, user_id, created_at, last_active) VALUES (%s, %s, NOW(), NOW())", (session_id, user_id))
                ip_address = request.remote_addr
                cur.execute("INSERT INTO user_login_history (user_id, ip_address, successful) VALUES (%s, %s, %s)", (user_id, ip_address, 1))
                cur.execute("INSERT INTO logs (user_id, activity_type, description) VALUES (%s, %s, %s)", (user_id, 'login', f'{username} logged in'))
                mysql.connection.commit()
                session['session_id'] = session_id
                
                flash('You are now logged in', 'success')
                cur.close()
                return redirect(url_for('index'))
            else:
                flash('Invalid password', 'danger')
        else:
            flash('User not registered', 'danger')
        cur.close()
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    if 'user_id' in session:
        user_id = session['user_id']
        username = session['username']
        
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO logs (user_id, activity_type, description) VALUES (%s, %s, %s)", (user_id, 'logout', f'{username} logged out'))
        ip_address = request.remote_addr
        cur.execute("INSERT INTO user_login_history (user_id, ip_address, successful) VALUES (%s, %s, %s)", (user_id, ip_address, 0))
        mysql.connection.commit()
        cur.close()

    if 'session_id' in session:
        session_id = session['session_id']
        cur = mysql.connection.cursor()
        cur.execute("UPDATE sessions SET last_active = NOW() WHERE session_id = %s", (session_id,))
        mysql.connection.commit()
        cur.close()
        
    session.clear()
    flash('You have been successfully logged out!', 'success')
    return redirect(url_for('login'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'email' not in session:
        flash('Please log in to view your profile.', 'danger')
        return redirect(url_for('login'))

    cur = mysql.connection.cursor()
    cur.execute("SELECT id, username, email, profile_picture FROM users WHERE email = %s", [session['email']])
    user = cur.fetchone()
    cur.close()

    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('logout'))

    user_data = {
        'id': user[0],
        'username': user[1],
        'email': user[2],
        'profile_picture': user[3]
    }

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        profile_picture = request.files.get('profile_picture')

        cur = mysql.connection.cursor()
        cur.execute("UPDATE users SET username = %s, email = %s WHERE email = %s", [username, email, session['email']])
        mysql.connection.commit()

        if profile_picture:
            # Ensure the upload directory exists
            upload_folder = os.path.join(app.root_path, 'static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            
            image_filename = secure_filename(profile_picture.filename)
            # Save using absolute path
            profile_picture.save(os.path.join(upload_folder, image_filename))
            
            # Store relative path in DB (for url_for)
            db_path = f'uploads/{image_filename}'
            cur.execute("UPDATE users SET profile_picture = %s WHERE email = %s", [db_path, session['email']])
            mysql.connection.commit()

        cur.close()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))


    return render_template('profile.html', user=user_data)

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html', username=session['username'])
    flash('Please log in first', 'danger')
    return redirect(url_for('login'))

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'username' in session:
        if request.method == 'POST':
            selected_theme = request.form.get('theme')
            session['theme'] = selected_theme

            cur = mysql.connection.cursor()
            cur.execute("SELECT settings_user_id FROM settings WHERE settings_user_id = (SELECT id FROM users WHERE email = %s)", [session['email']])
            existing_settings = cur.fetchone()

            if existing_settings:
                cur.execute("UPDATE settings SET dark_mode = %s WHERE settings_user_id = %s", [selected_theme, existing_settings[0]])
            else:
                cur.execute("INSERT INTO settings (settings_user_id, dark_mode) VALUES ((SELECT id FROM users WHERE email = %s), %s)", [session['email'], selected_theme])
            mysql.connection.commit()
            cur.close()

            flash(f'Theme changed to {selected_theme}!', 'success')
            return redirect(url_for('settings'))
        
        cur = mysql.connection.cursor()
        cur.execute("SELECT dark_mode FROM settings WHERE settings_user_id = (SELECT id FROM users WHERE email = %s)", [session['email']])
        theme = cur.fetchone()
        cur.close()

        current_theme = theme[0] if theme else session.get('theme', 'default')

        return render_template('settings.html', username=session['username'], theme=current_theme)

    flash('Please log in first', 'danger')
    return redirect(url_for('login'))

@app.before_request
def before_request():
    g.language = session.get('language', 'en')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if 'username' not in session:
        flash('Please log in first', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        feedback_text = request.form.get('feedback')
        rating = request.form.get('rating')
        
        if not feedback_text:
            flash('Feedback cannot be empty!', 'danger')
            return redirect(url_for('feedback'))
        
        if rating == '':
            rating = None
            
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO feedback(feedback_user_id, message, rating) VALUES(%s, %s, %s)", (session['user_id'], feedback_text, rating))
        mysql.connection.commit()
        cur.close()

        flash('Your feedback has been submitted!', 'success')
        return redirect(url_for('feedback'))

    return render_template('feedback.html', username=session['username'])

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if 'username' not in session:
        flash('Please log in first', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        message = request.form['message']
        username = session['username']
        
        cur = mysql.connection.cursor()
        cur.execute("SELECT id FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        
        if user:
            user_id = user[0]
            cur.execute("INSERT INTO contact (customer_id, message) VALUES (%s, %s)", (user_id, message))
            mysql.connection.commit()
            flash('Your message has been sent!', 'success')
        else:
            flash('User not found. Please log in again.', 'danger')

        cur.close()
        return redirect(url_for('contact'))

    return render_template('contact.html', username=session['username'])

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/animation_home')
def animation_home():
    return render_template('animation_home.html')

@app.route('/animation', methods=['GET', 'POST'])
def animation():
    if 'username' not in session:
        flash('Please log in to use the animation feature.', 'danger')
        return redirect(url_for('login'))

    text = ""
    words = []

    if request.method == 'POST':
        text = request.form.get('sen', '')
        remove_stopwords = request.form.get('remove_stopwords') == 'on'
        words = process_text_for_animation(text, remove_stopwords=remove_stopwords)

    return render_template('animation.html', words=words, text=text)


@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400

    try:
        import speech_recognition as sr
    except ImportError:
        return jsonify({
            "error": "speech_recognition is not installed. Install with: pip install SpeechRecognition"
        }), 501

    filename = audio_file.filename or "speech.wav"
    suffix = Path(filename).suffix.lower() or ".wav"
    input_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            input_path = Path(tmp_file.name)
            audio_file.save(str(input_path))

        if suffix != ".wav":
            return jsonify({
                "error": "Unsupported format. Upload WAV audio only."
            }), 415

        offline_text = _transcribe_with_vosk(input_path)
        if offline_text:
            return jsonify({"text": offline_text, "engine": "vosk"})

        recognizer = sr.Recognizer()
        with sr.AudioFile(str(input_path)) as source:
            audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_sphinx(audio_data)
            if text:
                return jsonify({"text": text, "engine": "sphinx"})
        except Exception:
            pass

        # Final fallback: online recognizer.
        text = recognizer.recognize_google(audio_data)
        return jsonify({"text": text, "engine": "google"})

    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand the audio"}), 422
    except sr.RequestError as exc:
        return jsonify({"error": f"Speech service is unavailable: {exc}"}), 503
    except Exception as exc:
        return jsonify({"error": f"Speech processing failed: {exc}"}), 500
    finally:
        if input_path and input_path.exists():
            try:
                input_path.unlink()
            except OSError:
                pass


def _transcribe_with_vosk(wav_path):
    model_path = os.getenv("VOSK_MODEL_PATH", "").strip()
    if not model_path:
        return None

    try:
        import vosk
    except ImportError:
        return None

    model_dir = Path(model_path)
    if not model_dir.exists():
        return None

    try:
        with wave.open(str(wav_path), "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                return None

            rec = vosk.KaldiRecognizer(vosk.Model(str(model_dir)), wf.getframerate())
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)

            final_result = json.loads(rec.FinalResult())
            text = final_result.get("text", "").strip()
            return text or None
    except Exception:
        return None


# Video Stream Routes
MODEL_CHECKPOINT = Path(os.getenv("GESTURA_MODEL_PATH", "Model/artifacts/gesture_transformer_126.pth"))
MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAMERA_INDEX = int(os.getenv("GESTURA_CAMERA_INDEX", "0"))
CAMERA_INDEXES_ENV = os.getenv("GESTURA_CAMERA_INDEXES", "")
CAMERA_PROBE_MAX = int(os.getenv("GESTURA_CAMERA_PROBE_MAX", "5"))
CAMERA_PROBE_ALL = os.getenv("GESTURA_CAMERA_PROBE_ALL", "0").strip().lower() in {"1", "true", "yes", "on"}
CAMERA_BACKENDS_ENV = os.getenv("GESTURA_CAMERA_BACKENDS", "").strip()


def _parse_camera_indices():
    indices = []
    if CAMERA_INDEXES_ENV.strip():
        for raw in CAMERA_INDEXES_ENV.split(","):
            token = raw.strip()
            if token.isdigit():
                idx = int(token)
                if idx not in indices:
                    indices.append(idx)

    if not indices:
        indices = [CAMERA_INDEX]
    elif CAMERA_INDEX not in indices:
        indices.insert(0, CAMERA_INDEX)

    if CAMERA_PROBE_ALL:
        for idx in range(CAMERA_PROBE_MAX):
            if idx not in indices:
                indices.append(idx)

    return indices


def _parse_camera_backends_env():
    if not CAMERA_BACKENDS_ENV:
        return None

    name_to_backend = {
        "AUTO": None,
        "ANY": cv2.CAP_ANY,
        "DSHOW": cv2.CAP_DSHOW,
        "MSMF": cv2.CAP_MSMF,
    }
    result = []
    for raw in CAMERA_BACKENDS_ENV.split(","):
        key = raw.strip().upper()
        if key in name_to_backend:
            result.append((key, name_to_backend[key]))
    return result if result else None


def _camera_backends():
    env_backends = _parse_camera_backends_env()
    if env_backends:
        return env_backends

    if os.name == "nt":
        return [
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
        ]
    return [("ANY", cv2.CAP_ANY)]


def _open_working_camera():
    attempts = []
    indices = _parse_camera_indices()

    for index in indices:
        for backend_name, backend in _camera_backends():
            attempts.append(f"{index}:{backend_name}")
            cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            # Warm up briefly to verify the device can actually produce frames.
            frame_ok = False
            for _ in range(20):
                ret, _frame = cap.read()
                if ret:
                    frame_ok = True
                    break
                time.sleep(0.05)

            if frame_ok:
                return cap, index, backend_name, attempts

            cap.release()

    return None, None, None, attempts


def load_gesture_model():
    if not MODEL_CHECKPOINT.exists():
        print(f"Model checkpoint not found: {MODEL_CHECKPOINT}")
        return None, []

    try:
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location=MODEL_DEVICE)
        label_map = list(checkpoint.get("label_map", []))
        model = GestureTransformer(
            input_dim=int(checkpoint.get("feature_size", FEATURE_SIZE)),
            seq_length=int(checkpoint.get("sequence_length", SEQUENCE_LENGTH)),
            num_classes=int(checkpoint.get("num_classes", max(len(label_map), 1))),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(MODEL_DEVICE)
        model.eval()
        return model, label_map
    except Exception as exc:
        print(f"Error loading gesture model: {exc}")
        return None, []


gesture_model, gesture_labels = load_gesture_model()


def _error_frame(message: str, width: int = 960, height: int = 540) -> bytes:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "Sign-to-text stream unavailable",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message[:95],
        (30, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return b""
    return buffer.tobytes()


def _yield_error_stream(message: str):
    frame = _error_frame(message)
    if not frame:
        return
    while True:
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.5)

CAMERA_STREAM_LOCK = threading.Lock()


@app.route('/generate_frames')
def generate_frames():
    if not CAMERA_STREAM_LOCK.acquire(blocking=False):
        message = "Camera stream already active in another request. Close other tabs and retry."
        print(f"Error: {message}")
        yield from _yield_error_stream(message)
        return

    try:
        from mediapipe import solutions as mp_solutions
    except ImportError:
        message = "mediapipe import failed. Install compatible wheel for this Python version."
        print(f"Error: {message}")
        yield from _yield_error_stream(message)
        CAMERA_STREAM_LOCK.release()
        return

    cap, active_index, active_backend, attempts = _open_working_camera()
    if cap is None:
        message = (
            "Cannot open camera. Tried "
            + ", ".join(attempts[:8])
            + ". Set GESTURA_CAMERA_INDEX / GESTURA_CAMERA_INDEXES (and optionally GESTURA_CAMERA_BACKENDS)."
        )
        print(f"Error: {message}")
        yield from _yield_error_stream(message)
        CAMERA_STREAM_LOCK.release()
        return
    print(f"Camera opened at index {active_index} using backend {active_backend}")

    mp_hands = mp_solutions.hands
    mp_drawing = mp_solutions.drawing_utils
    hands = None
    try:
        hands = mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2,
        )
    except Exception as exc:
        message = f"MediaPipe Hands init failed: {exc}"
        print(f"Error: {message}")
        yield from _yield_error_stream(message)
        cap.release()
        CAMERA_STREAM_LOCK.release()
        return

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    prediction_text = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap, active_index, active_backend, _ = _open_working_camera()
                if cap is None:
                    message = "Camera disconnected during stream."
                    print(f"Error: {message}")
                    yield from _yield_error_stream(message)
                    return
                print(f"Camera reconnected at index {active_index} using backend {active_backend}")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            sequence.append(extract_landmark_features(results))

            if gesture_model is not None and len(sequence) == SEQUENCE_LENGTH:
                input_np = normalize_sequence(sequence)
                input_tensor = torch.tensor(input_np, dtype=torch.float32, device=MODEL_DEVICE).unsqueeze(0)
                with torch.no_grad():
                    logits = gesture_model(input_tensor)
                    pred_idx = int(torch.argmax(logits, dim=1).item())

                if gesture_labels and pred_idx < len(gesture_labels):
                    prediction_text = str(gesture_labels[pred_idx])
                else:
                    prediction_text = str(pred_idx)

            if prediction_text:
                cv2.putText(
                    frame,
                    f"Prediction: {prediction_text}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            elif gesture_model is None:
                cv2.putText(
                    frame,
                    "Model missing: train and place checkpoint at Model/artifacts/gesture_transformer.pth",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            ok, buffer = cv2.imencode('.jpg', frame)
            if not ok:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        if hands is not None:
            hands.close()
        CAMERA_STREAM_LOCK.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera_status')
def camera_status():
    if CAMERA_STREAM_LOCK.locked():
        return jsonify({
            "available": True,
            "in_use": True,
            "detail": "Camera stream already active.",
        })

    cap, active_index, active_backend, attempts = _open_working_camera()
    if cap is None:
        return jsonify({
            "available": False,
            "attempts": attempts[:18],
            "detail": "OpenCV could not open any camera device.",
        }), 503

    cap.release()
    return jsonify({
        "available": True,
        "in_use": False,
        "index": active_index,
        "backend": active_backend,
        "attempts": attempts[:18],
        "detail": "Camera is available to OpenCV.",
    })


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
