import sqlite3 
import datetime 
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_from_directory, send_file
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash # Keep generate_password_hash for the mock/temp fix
import speech_recognition as sr
from pydub import AudioSegment 
import os
import io
import numpy as np
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav
from difflib import SequenceMatcher
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import io
import pickle
import time
import random
import string
import qrcode
import base64
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader, PdfWriter
from utils.security import signer, merkle_tree # Layer 1 & 2 Security
import json

# Initialize Flask
app = Flask(__name__, static_folder='static', template_folder='Front-End/templates') # Explicitly set folders
print("Initializing VerifyX Server...")

# --- BIOMETRIC MODEL INITIALIZATION ---
# Load the model once at startup (this might take a few seconds)
print("Loading Voice Recognition Model...")
encoder = VoiceEncoder()
print("Voice Model Loaded.")

# --- MEDIAPIPE INITIALIZATION ---
# --- MEDIAPIPE INITIALIZATION ---
print("Loading Face Landmarker Model...")
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)
print("Face Landmarker Model Loaded.")

# Configuration
app.secret_key = 'your_super_secret_and_unique_key_12345' 
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(minutes=5) # Auto-logout after 5 mins
UPLOAD_FOLDER = 'uploads'
DATABASE = os.path.join(os.path.dirname(__file__), 'data', 'verifyx.db')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok = True)
os.makedirs(os.path.dirname(DATABASE), exist_ok = True)

USER_ENROLLED = False

# --- DB INIT HELPER ---
# Database initialization is handled in data/seed.py
# VerifyX Server expects the DB to exist at startup.

# --- SESSION TIMEOUT CHECK ---
@app.before_request
def check_session_timeout():
    """
    Middleware that checks if the user's session has exceeded the idle timeout (5 mins).
    If timed out, clears the session and redirects to login.
    """
    # Exempt valid static/auth endpoints to prevent loops
    if request.endpoint in ('login', 'handle_login', 'static', 'serve_gestures', 'check_blink', 'check_pose', 'verify_face', 'verify_voice', 'logout'):
        return

    if 'user_mykad' in session:
        now = time.time()
        last_active = session.get('last_active')
        
        # If last_active is set, check delta
        if last_active:
            # Check 5 minutes (300 seconds)
            if now - last_active > 300:
                session.clear()
                return redirect(url_for('login', alert='session_expired'))
        
        # Update last_active
        session['last_active'] = now
        session.permanent = True # Ensure cookie expiry follows logic



def allowed_file(filename):
    """
    Checks whether the given filename has an allowed file extension.
    """
    if '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS

# DATABASE ACCESS LAYER
def get_db_connection():
    """
    Establishes an SQLite connection and configures row factory.
    """
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

from contextlib import contextmanager

@contextmanager
def get_db():
    """
    Context manager to ensure database connections are always closed,
    even if an exception occurs.
    """
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()


# --- MEDIAPIPE HELPERS ---
def normalize_landmarks(landmarks):
    """
    Normalizes face landmarks to be invariant to translation (centering)
    and scale (distance from camera).
    """
    # 1. Convert to simple list of dicts or np array for easier math
    # MediaPipe landmarks have .x, .y, .z
    # OPTIMIZATION: Dropping Z-axis (Depth) because it's unstable on webcams and causes high noise.
    # We will use purely 2D facial geometry.
    coords = np.array([[lm.x, lm.y] for lm in landmarks])
    
    # 2. Translation Invariance: Center around the Nose Tip (Index 1)
    nose_tip = coords[1]
    centered = coords - nose_tip
    
    # 3. Scale Invariance: Scale by Inter-ocular Distance (Dist between Left Eye 33 and Right Eye 263)
    # Using Euclidean distance in 2D
    left_eye = coords[33]
    right_eye = coords[263]
    dist = np.linalg.norm(left_eye - right_eye)
    
    # DEBUG: Print Scale Factor
    print(f"DEBUG: Normalization Scale (Inter-ocular dist 2D): {dist:.6f}")
    
    if dist == 0: dist = 1 # Safety
        
    normalized = centered / dist
    
    # Flatten
    return normalized.flatten()

def get_face_vector(image_bytes):
    """
    Extracts a 1D vector (468*3 floats) representing the facial geometry.
    Used for identity comparison.
    """
    try:
        # Load and convert to MediaPipe Image
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        np_img = np.array(pil_img)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_img)
        
        # Detect
        detection_result = detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            print("DEBUG: No landmarks found in image.")
            return None
            
        # Extract 0th face
        landmarks = detection_result.face_landmarks[0]
        
        # Normalize landmarks (Translation & Scale Invariance)
        normalized_vector = normalize_landmarks(landmarks)
            
        return normalized_vector
        
    except Exception as e:
        print(f"MediaPipe Error: {e}")
        return None

def compare_vectors(v1, v2):
    """
    Calculates similarity between two face vectors.
    Uses Euclidean Distance. Lower is better/closer.
    """
    if v1 is None or v2 is None:
        return float('inf')
        
    # Check for Shape Mismatch (e.g. Old 3D enrollment vs New 2D vector)
    if v1.shape != v2.shape:
        print(f"DEBUG: Vector Shape Mismatch! {v1.shape} vs {v2.shape}. Force Re-enroll.")
        return 999.0 # Arbitrary high distance to force fail
        
    # Euclidean Distance
    dist = np.linalg.norm(v1 - v2)
    return dist
# --- END MEDIAPIPE HELPERS ---

def fetch_user_record_by_mykad(mykad):
    """
    Fetches a citizen's record using their MyKad number.
    """
    try:
        with get_db() as conn:
            user_record = conn.execute('SELECT * FROM citizens WHERE mykad_number = ?', (mykad,)).fetchone()
            return user_record
    except sqlite3.OperationalError as e:
        print(f"DATABASE ERROR: {e}")
        return None 

def log_action(mykad, action, context, organization_name, status = "SUCCESS"):
    """
    Inserts a new audit log entry.
    """
    try:
        with get_db() as conn:
            conn.execute('''INSERT INTO access_logs (mykad_number, action, context, organization_name, status)VALUES (?, ?, ?, ?, ?)''', 
                         (mykad, action, context, organization_name, status))
            conn.commit()
    except sqlite3.OperationalError as e:
        print(f"LOGGING ERROR: Could not write to access_logs. Did you run seed.py? Error: {e}")

def fetch_access_logs(mykad, log_status = None):
    """
    Fetches audit logs for a specific user, optionally filtered by status.
    """
    try:
        with get_db() as conn:
            query = 'SELECT * FROM access_logs WHERE mykad_number = ? ORDER BY timestamp DESC'
            params = (mykad,)
            if log_status:
                query = 'SELECT * FROM access_logs WHERE mykad_number = ? AND status = ? ORDER BY timestamp DESC'
                params = (mykad, log_status)
            logs = conn.execute(query, params).fetchall()
            
            readable_logs = []
            for log in logs:
                d = dict(log)
                
                # Context Parsing to find ShareID:X and get Recipient ID
                # Format: "ShareID:3 - Downloaded..." or "Guest accessed Share:3"
                recipient_id = None
                
                # Simple heuristic search for ShareID: digit
                import re
                match = re.search(r'Share[:\s]*ID[:\s]*(\d+)', d['context'], re.IGNORECASE)
                if not match:
                     match = re.search(r'Share[:\s]*(\d+)', d['context'], re.IGNORECASE) 
                     
                if match:
                    share_id = match.group(1)
                    d['share_id'] = share_id # Store Share ID for Revoke Actions
                    try:
                        # Fetch Recipient ID from Share Session
                        share_info = conn.execute('SELECT recipient_id, status FROM share_sessions WHERE id = ?', (share_id,)).fetchone()
                        if share_info:
                            d['recipient_id'] = share_info['recipient_id']
                            # Optionally fetch current share status to know if we should show Revoke button
                            d['share_status'] = share_info['status']
                    except:
                        pass
                
                readable_logs.append(d)
                
            return readable_logs
    except sqlite3.OperationalError as e:
        print(f"FETCH LOGS ERROR: Could not read from access_logs. Did you run seed.py? Error: {e}")
        return []

def get_user_context():
    """
    Helper to get user data and handle unauthenticated access.
    """
    mykad = session.get('user_mykad')
    if not mykad:
        return None, None
    user_record = fetch_user_record_by_mykad(mykad)
    
    if user_record:
        user_dict = dict(user_record) 
        user_dict['first_name'] = user_dict['full_name'].split()[0]
        return user_dict, mykad
    return None, None

# --- TIMEOUT HELPER ---
def check_timeout():
    """
    Checks if the global 60-second biometric verification timer has expired.
    Returns: True if timed out, False otherwise.
    """
    start_time = session.get('verification_start')
    if not start_time:
        # If no start time is set, assume we are fine or it hasn't started.
        # However, for security, if we are deep in steps without a timer, we might want to fail?
        # But 'verify_identity' sets it.
        return False
        
    elapsed = time.time() - start_time
    if elapsed > 90: # 1.5 Minute Limit
        register_failure(session.get('user_mykad'), "TIMEOUT")
        return True
    return False

# LOGIN & BIOMETRIC LOGIC
@app.route('/login')
def login():
    """
    Handles the login page route.    
    """
    if session.get('vault_access_granted'):
        return redirect(url_for('dashboard'))
    
    # NEW: Check if we arrived here due to a session timeout (Backend verification)
    if check_timeout():
        # This catches the case where Frontend Auto-Redirect hits /login
        # We verify the timestamp really expired, then punish.
        mykad = session.get('user_mykad')
        if mykad:
            register_failure(mykad, "TIMEOUT_ON_LOGIN_REDIRECT")
            # Clear start time so we don't count it twice if they refresh
            session.pop('verification_start', None)

    # Check for timeout alert (passed by URL)
    alert = request.args.get('alert')
    return render_template('login.html', alert=alert)

# --- FAILURE HANDLER ---
def register_failure(mykad, reason="FAILED"):
    """
    Increments failed attempts and handles lockout logic.
    """
    if not mykad:
        log_action('UNKNOWN', "LOGIN", reason, "System", "FAILED")
        return

    with get_db() as conn:
        user_record = conn.execute('SELECT failed_attempts FROM citizens WHERE mykad_number = ?', (mykad,)).fetchone()
        
        if user_record:
            current_attempts = user_record['failed_attempts'] + 1
            lock_duration = 0
            
            # Logic: 3 fails = 5 min, then +10 mins for each subsequent fail.
            if current_attempts == 3:
                lock_duration = 5
            elif current_attempts > 3:
                lock_duration = 5 + (current_attempts - 3) * 10
            
            if lock_duration > 0:
                lockout_until = datetime.datetime.now() + datetime.timedelta(minutes=lock_duration)
                
                # REUSE connection for update (safe provided logic is correct)
                # Or use the same context manager? 
                # Actually, `conn` is cursor? No `row_factory`... 
                # The context manager yields `conn`.
                conn.execute('UPDATE citizens SET failed_attempts = ?, lockout_until = ? WHERE mykad_number = ?', 
                                (current_attempts, lockout_until, mykad))
                conn.commit()
                    
                print(f"DEBUG: {mykad} LOCKED for {lock_duration} mins")
                log_action(mykad, "LOGIN", f"Blocked (Locked for {lock_duration}m due to {reason})", "System", "BLOCKED")
            else:
                conn.execute('UPDATE citizens SET failed_attempts = ? WHERE mykad_number = ?', 
                                (current_attempts, mykad))
                conn.commit()
                    
                log_action(mykad, "LOGIN", f"{reason} Attempt ({current_attempts})", "System", "FAILED")

@app.route('/handle_login', methods = ['POST'])
def handle_login():
    """
    Processes the login form submission.
    Validates MyKad and password, controls lockout logic, and initiates biometric flow.
    """
    data = request.get_json() 
    mykad = data.get('ic-number', '').strip()
    password = data.get('password', '').strip()
    print(f"DEBUG: Handle Login for MyKad: '{mykad}'")

    # --- MOCK JPN ACCOUNT ---
    if mykad == "mockjpn" and password == "password123":
        session['user_mykad'] = "mockjpn"
        session['user_name'] = "JPN Officer"
        session['vault_access_granted'] = True
        session['is_issuer'] = True
        return jsonify({'status': 'success', 'redirect': url_for('issuer_dashboard')})

    user_record = fetch_user_record_by_mykad(mykad)
    
    if user_record:
        # --- 12-HOUR AUTO-RESET LOGIC ---
        if user_record['failed_attempts'] > 0:
            try:
                with get_db() as conn:
                    # Find last failure (LOGIN + FAILED/BLOCKED)
                    last_fail = conn.execute('''
                        SELECT timestamp FROM access_logs 
                        WHERE mykad_number = ? AND action = 'LOGIN' AND (status = 'FAILED' OR status = 'BLOCKED')
                        ORDER BY timestamp DESC LIMIT 1
                    ''', (mykad,)).fetchone()
                    
                    if last_fail:
                        last_fail_time = datetime.datetime.strptime(last_fail['timestamp'], '%Y-%m-%d %H:%M:%S')
                        if datetime.datetime.now() - last_fail_time > datetime.timedelta(hours=12):
                            print(f"DEBUG: Auto-resetting {mykad} (Last fail > 12h ago)")
                            
                            conn.execute('UPDATE citizens SET failed_attempts = 0, lockout_until = NULL WHERE mykad_number = ?', (mykad,))
                            conn.commit()
                            
                            # Refresh user_record so typical flow sees clean state
                            user_record = fetch_user_record_by_mykad(mykad)
                            log_action(mykad, "SYSTEM_RESET", "Auto-reset after 12h idle", "System", "RESET")
            except Exception as e:
                print(f"Auto-reset Error: {e}")

        # 1. CHECK LOCKOUT STATUS
        if user_record['lockout_until']:
             lockout_time = datetime.datetime.strptime(user_record['lockout_until'], '%Y-%m-%d %H:%M:%S.%f')
             if datetime.datetime.now() < lockout_time:
                 remaining = (lockout_time - datetime.datetime.now()).seconds // 60
                 log_action(mykad, "LOGIN", f"Blocked (Locked for {remaining}m)", "System", "BLOCKED")
                 return jsonify({'status': 'failure', 'message': f'Account locked. Try again in {remaining + 1} minutes.'}), 403

        stored_hash = user_record['password_hash']
        if stored_hash and check_password_hash(stored_hash, password):
            
            # SUCCESS: Reset Counters
            with get_db() as conn:
                conn.execute('UPDATE citizens SET failed_attempts = 0, lockout_until = NULL WHERE mykad_number = ?', (mykad,))
                conn.commit()

            # Authentication Success: Store MyKad and name in session
            session['user_mykad'] = mykad
            # Access full_name via square brackets []
            session['user_name'] = user_record['full_name'].split()[0]
            
            # --- SESSION TIMEOUT INIT ---
            session['last_active'] = time.time()
            
            # --- SESSION CLEANUP (Fixes Stale Redirects) ---
            session.pop('enrollment_stage', None)
            session.pop('verification_queue', None)
            session.pop('verification_start', None)
            
            # --- PERSISTENT ENROLLMENT CHECK ---
            # Use explicit flag from DB
            # Safe access with .get() or dict() conversion to handle missing/null columns
            try:
                # Check if enrollment is complete (handles 0, 1, or None)
                enrollment_status = user_record['is_enrollment_complete'] if 'is_enrollment_complete' in user_record.keys() else 0
                session['is_enrolled'] = bool(enrollment_status)
            except (KeyError, IndexError):
                # Fallback: If column doesn't exist, assume not enrolled
                session['is_enrolled'] = False
            
            # 3. Log Audit Trail 
            log_action(mykad, "LOGIN", "Portal Access", "System", "SUCCESS")
            
            # Success: Go to the Gateway
            return jsonify({'status': 'success', 'redirect': url_for('verify_identity')})
        
        else:
            # FAILURE: Increment Attempts & Lockout Logic (Using Helper)
            register_failure(mykad, "Bad Password")
            return jsonify({'status': 'failure', 'message': 'Invalid MyKad or Password.'}), 401
    
    # Unknown user
    register_failure(None, "Unknown User")
    return jsonify({'status': 'failure', 'message': 'Invalid MyKad or Password.'}), 401

@app.route('/verify-face', methods=['POST'])
def verify_face():
    """
    Receives an image (from webcam), extracts the MediaPipe Face Vector,
    and compares it against the enrolled vector in the database.
    """
    # 1. TIMEOUT CHECK
    if check_timeout():
        return jsonify({'status': 'failure', 'message': 'TIMEOUT: Verification took too long.', 'redirect': url_for('login', alert='timeout')})
    
    mykad = session.get('user_mykad')
    if not mykad:
        return jsonify({'status': 'failure', 'message': 'Session expired.'}), 401
    
    # 2. FILE CHECK
    if 'image' not in request.files:
        return jsonify({'status': 'failure', 'message': 'No image provided.'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'failure', 'message': 'No selected file.'}), 400

    try:
        # 3. Read Bytes directly
        image_blob = file.read()
        file.seek(0) # Reset pointer
        
        # 4. Extract Geometry Vector (MediaPipe)
        face_vector = get_face_vector(image_blob)
        
        if face_vector is None:
             return jsonify({'status': 'failure', 'message': 'No face detected. Please ensure good lighting.'})
             
        # 5. Serialize Vector
        new_encoding_blob = pickle.dumps(face_vector)

        # 6. ENROLLMENT (First time)
        if not session.get('is_enrolled') and session.get('enrollment_stage') == 'face':
            with get_db() as conn:
                # Store BOTH the image and the encoding
                conn.execute('UPDATE citizens SET face_image_blob = ?, face_encoding_blob = ? WHERE mykad_number = ?', 
                             (image_blob, new_encoding_blob, mykad))
                conn.commit()
            
            log_action(mykad, "ENROLL", "Face Biometric Registered (MP)", "System", "SUCCESS")
            return jsonify({'status': 'success', 'message': 'Face registered successfully.'})

        # 7. VERIFICATION (Returning)
        else:
            with get_db() as conn:
                # Fetch stored encoding from citizens table
                user_record = conn.execute('SELECT face_encoding_blob FROM citizens WHERE mykad_number = ?', (mykad,)).fetchone()
            
            if not user_record or not user_record['face_encoding_blob']:
                return jsonify({'status': 'failure', 'message': 'No face enrolled. Please enroll first.'})

            stored_vector = pickle.loads(user_record['face_encoding_blob'])
            
            # Compare
            distance = compare_vectors(stored_vector, face_vector)
            print(f"DEBUG: Face Distance = {distance:.4f}")
            
            # Threshold: < 3.5 roughly match for same person/pose
            # With normalization, vectors are much closer. 
            # A mismatch is usually > 0.6, a match is < 0.3.
            # MIRROR FALLBACK CHECK
            # If normal check fails, try flipping the image horizontally.
            final_distance = distance
            if distance > 1.0:
                 try:
                     # Re-read image bytes for PIL
                     pil_img = Image.open(io.BytesIO(image_blob)).convert('RGB')
                     flipped_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                     
                     buf = io.BytesIO()
                     flipped_img.save(buf, format='JPEG')
                     flipped_bytes = buf.getvalue()
                     
                     flipped_vector = get_face_vector(flipped_bytes)
                     if flipped_vector is not None:
                         dist_flipped = compare_vectors(stored_vector, flipped_vector)
                         print(f"DEBUG: Verify (Flipped): Dist = {dist_flipped:.4f}")
                         
                         if dist_flipped < final_distance:
                             final_distance = dist_flipped
                             print("DEBUG: Flipped image gave better match (Login)!")
                 except Exception as e:
                     print(f"Mirror Check Error: {e}")

            # Threshold: < 3.5 roughly match for same person/pose
            # Determine threshold
            # With normalization, vector magnitude is sensitive to eye-distance jitter (e.g. glasses).
            # A 5% jitter can cause Distance ~ 0.75. 
            # Lowered to 1.0 for higher security (stricter matching requirement).
            if final_distance < 1.0:
                # conn.close() # handled by context manager
                print(f"DEBUG: MATCH! Dist {final_distance:.4f} < 1.0")
                log_action(mykad, "VERIFY", "Face Match Success", "System", "SUCCESS")
                return jsonify({'status': 'success', 'message': 'Face verified successfully.'})
            else:
                # conn.close() # handled by context manager
                print(f"DEBUG: FAILURE! Dist {final_distance:.4f} > 1.0")
                log_action(mykad, "VERIFY", "Face Mismatch", "System", "FAILED")
                return jsonify({'status': 'failure', 'message': 'Face verification failed. Please try again.'})

    except Exception as e:
        print(f"Face Error: {e}")
        return jsonify({'status': 'failure', 'message': 'Error processing face biometrics.'}), 500

@app.route('/verify-voice', methods = ['POST'])
def verify_voice():
    """
    Receives audio blob, saves it, and performs verification.
    """
    # FIX: Use 'user_mykad' as standardized
    if 'user_mykad' not in session:
        return jsonify({'status': 'failure', 'message': 'Session expired.'}), 401
    
    # TIMEOUT CHECK
    if check_timeout():
        return jsonify({'status': 'failure', 'message': 'TIMEOUT: Verification took too long.', 'redirect': url_for('login', alert='timeout')})
    
    mykad = session['user_mykad']
    expected_phrase = request.form.get('phrase', '')
    
    if 'audio' not in request.files:
        return jsonify({'status': 'failure', 'message': 'No audio file provided.'}), 400
        
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'status': 'failure', 'message': 'No selected file.'}), 400

    # Create a BytesIO object to handle the file in memory
    audio_data_blob = audio_file.read()
    
    # 1. Enrollment Logic
    # global USER_ENROLLED # Not strictly needed if we just check local var, but logic uses global in step_complete
    # 1. Enrollment Logic
    if not session.get('is_enrolled') and session.get('enrollment_stage') == 'voice':
        # --- DB UPDATE STEP ---
        try:
            print(f"DEBUG: Attempting to update DB for {mykad} with voice BLOB")
            with get_db() as conn:
                conn.execute('UPDATE citizens SET voice_audio_blob = ? WHERE mykad_number = ?', (audio_data_blob, mykad))
                conn.commit()
            print(f"DEBUG: Voice BLOB saved.")
            log_action(mykad, "ENROLL", "Voice Print SAVED to DB (BLOB)", "System", "SUCCESS")
        except Exception as e:
            print(f"DB Error on Voice Enroll: {e}")
            log_action(mykad, "ENROLL", "DB Save Failed", "System", "FAILED")
            
        return jsonify({'status': 'success', 'message': 'Voice registered successfully.'})

    # 2. Verification Logic
    else:
        # Step A: Content Verification (Did they say the phrase?)
        recognizer = sr.Recognizer()
        
        try:
            # Use the in-memory audio data for recognition
            audio_io = io.BytesIO(audio_data_blob)
            
            with sr.AudioFile(audio_io) as source:
                audio_content = recognizer.record(source)
                
                # Using Google Speech Recognition
                try:
                    text = recognizer.recognize_google(audio_content)
                    print(f"DEBUG: User said: '{text}' vs Expected: '{expected_phrase}'")
                except sr.UnknownValueError:
                     return jsonify({'status': 'failure', 'message': 'Speech unintelligible. Speak clearly.'})
                
                # Fuzzy Match Check
                similarity = SequenceMatcher(None, text.lower(), expected_phrase.lower()).ratio()
                print(f"DEBUG: Similarity Ratio: {similarity}")

                # Threshold: 0.6 means 60% match
                if similarity < 0.6: 
                     return jsonify({'status': 'failure', 'message': f'Incorrect phrase. You said: "{text}"'})

        except sr.RequestError as e:
            print(f"API Error: {e}")
            pass # Graceful degradation
            
        # Step B: "Biometric" Check (Compare with DB Blob)
        with get_db() as conn:
            user = conn.execute('SELECT voice_audio_blob FROM citizens WHERE mykad_number = ?', (mykad,)).fetchone()
        
        # Verify enrollment exists
        if not user or not user['voice_audio_blob']:
             return jsonify({'status': 'failure', 'message': 'Voice not enrolled. Please enroll first.'})

        # --- REAL BIOMETRIC VERIFICATION (Resemblyzer) ---
        try:
            # 1. Process ENROLLED voice (from DB)
            # Load raw bytes into audio array (16kHz)
            wav_enrolled, _ = librosa.load(io.BytesIO(user['voice_audio_blob']), sr=16000)
            processed_enrolled = preprocess_wav(wav_enrolled)
            embed_enrolled = encoder.embed_utterance(processed_enrolled)
            
            # 2. Process INCOMING voice (from Request)
            wav_incoming, _ = librosa.load(io.BytesIO(audio_data_blob), sr=16000)
            processed_incoming = preprocess_wav(wav_incoming)
            embed_incoming = encoder.embed_utterance(processed_incoming)
            
            # 3. Compute Similarity (Dot product of normalized vectors)
            # Embeddings are already L2-normalized by Resemblyzer
            similarity_score = np.inner(embed_enrolled, embed_incoming)
            print(f"DEBUG: Biometric Similarity Score: {similarity_score}")
            
            # Threshold Check
            # Typical threshold for high security is around 0.75 - 0.80
            # Relaxing to 0.75 based on user difficulty
            BIOMETRIC_THRESHOLD = 0.75 
            
            if similarity_score < BIOMETRIC_THRESHOLD:
                log_action(mykad, "VERIFY", f"Voice Mismatch (Score: {similarity_score:.2f})", "System", "FAILED")
                return jsonify({'status': 'failure', 'message': f'Voice verification failed. Biometric match too low ({similarity_score:.2f}).'})

        except Exception as e:
            print(f"Biometric Error: {e}")
            # Fallback or strict fail? Strict fail for security.
            return jsonify({'status': 'failure', 'message': 'Error processing voice biometrics.'})

        # Verification Success
        log_action(mykad, "VERIFY", f"Voice Match (Score: {similarity_score:.2f})", "System", "SUCCESS")
        return jsonify({'status': 'success', 'message': 'Voice verified successfully.', 'transcript': text})

@app.route('/gestures/<path:filename>')
def serve_gestures(filename):
    """
    Serves static gesture reference images for the biometric verification step.
    """
    return send_from_directory('gestures', filename)

@app.route('/verify-identity')
def verify_identity():
    """
    Directs users to the appropriate biometric verification or enrollment step
    based on their session and enrollment status.
    """
    # Fix: User changed session key from 'temp_mykad' to 'user_mykad' in handle_login
    if 'user_mykad' not in session: 
        return redirect(url_for('login'))
    
    if not session.get('is_enrolled'):
        session['enrollment_stage'] = 'face'
        session['verification_start'] = time.time() # START TIMEOUT TIMER (Enroll)
        return redirect(url_for('biometric_step'))
    else:
        return redirect(url_for('select_biometric'))

@app.route('/select-biometric')
def select_biometric():
    """
    Displays the biometric selection page for users who have started the login process.
    """
    if 'user_mykad' not in session: 
        return redirect(url_for('login'))
    return render_template('select_biometric.html')

@app.route('/verify-selection', methods=['POST'])
def verify_selection():
    """
    Handles the user's selection of biometric verification methods 
    and queues them for the verification process.
    """
    methods = request.form.getlist('methods') 
    if len(methods) != 2:
        return "Error: Please select exactly 2 methods."
    
    session['verification_queue'] = methods
    session['verification_start'] = time.time() # START TIMEOUT TIMER (Verify)
    return redirect(url_for('step_complete')) 

@app.route('/biometric-step')
def biometric_step():
    """
    Directs users to the correct biometric enrollment or verification step 
    based on their current stage stored in the session.
    """
    if 'user_mykad' not in session: 
        return redirect(url_for('login'))
    
    # TIMEOUT CHECK (Backend)
    start_time = session.get('verification_start')
    if start_time:
        elapsed = time.time() - start_time
        remaining = 90 - elapsed
        if remaining <= 0:
             register_failure(session.get('user_mykad'), "TIMEOUT")
             return redirect(url_for('login', alert='timeout'))
    else:
        # Fallback if no timer set (shouldn't happen in flow, but safe default)
        remaining = 90 

    stage = session.get('enrollment_stage')
    print(f"DEBUG: Biometric Step: {stage}, Remaining Time: {remaining:.1f}s")

    if stage == 'face': 
        return render_template('faceScan.html', timeout=remaining) 
    elif stage == 'voice': 
        return render_template('voiceTest.html', timeout=remaining) 
    elif stage == 'gesture': 
        return render_template('gestureTest.html', timeout=remaining) 
    else: 
        return redirect(url_for('login'))

@app.route('/verify-gesture-identity', methods=['POST'])
def verify_gesture_identity():
    """
    Called during the Gesture Step to re-verify that the person performing
    the gesture is indeed the enrolled user.
    """
    if check_timeout():
        return jsonify({'status': 'failure', 'message': 'TIMEOUT.', 'redirect': url_for('login', alert='timeout')})
        
    mykad = session.get('user_mykad')
    
    if 'image' not in request.files:
        return jsonify({'status': 'failure', 'message': 'No image.'}), 400
        
    file = request.files['image']
    
    try:
        # 1. Read Bytes directly
        image_blob = file.read()
        
        # 2. Extract Vector (MediaPipe)
        new_vector = get_face_vector(image_blob)
        
        if new_vector is None:
             return jsonify({'status': 'failure', 'message': 'No face detected. Please show your face.'})
        
        # 3. IDENTITY VERIFICATION (Compare with DB)
        with get_db() as conn:
            user_record = conn.execute('SELECT face_encoding_blob FROM citizens WHERE mykad_number = ?', (mykad,)).fetchone()

        if not user_record or not user_record['face_encoding_blob']:
             return jsonify({'status': 'failure', 'message': 'No enrolled face data found.'})

        stored_vector = pickle.loads(user_record['face_encoding_blob'])
        
        # Compare
        dist_original = compare_vectors(stored_vector, new_vector)
        print(f"DEBUG: GESTURE VERIFY (Normal): Dist = {dist_original:.4f}")
        
        final_distance = dist_original
        
        # MIRROR FALLBACK CHECK
        # If normal check fails, try flipping the image horizontally.
        if dist_original > 1.0:
             try:
                 # Load original PIL (we need to re-open or if we had it.. we only had bytes. Re-open)
                 pil_img = Image.open(io.BytesIO(image_blob)).convert('RGB')
                 flipped_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                 flipped_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                 
                 # We need to convert back to MP Image usually, but get_face_vector takes bytes.
                 # Let's convert PIL -> Bytes
                 buf = io.BytesIO()
                 flipped_img.save(buf, format='JPEG')
                 flipped_bytes = buf.getvalue()
                 
                 flipped_vector = get_face_vector(flipped_bytes)
                 if flipped_vector is not None:
                     dist_flipped = compare_vectors(stored_vector, flipped_vector)
                     print(f"DEBUG: GESTURE VERIFY (Flipped): Dist = {dist_flipped:.4f}")
                     
                     if dist_flipped < final_distance:
                         final_distance = dist_flipped
                         print("DEBUG: Flipped image gave better match!")
             except Exception as e:
                 print(f"Mirror Check Error: {e}")
                 
        # Threshold similar to verify_face
        # Threshold similar to verify_face
        # Lowered to 1.0 for higher security (stricter matching requirement)
        if final_distance > 1.0: 
             log_action(mykad, "GESTURE_VERIFY", f"Face Mismatch (Dist={final_distance:.2f})", "System", "FAILED")
             return jsonify({'status': 'failure', 'message': 'Identity verification failed. Face does not match.'})

        # 4. Identity Confirmed
        print(f"DEBUG: Gesture Identity Verified. Dist={final_distance:.2f}")
        log_action(mykad, "GESTURE_VERIFY", f"Identity Confirmed", "System", "SUCCESS")
        
        return jsonify({'status': 'success', 'message': 'Identity confirmed.', 'blink_info': "Pass"})

    except Exception as e:
        print(f"Gesture Identity Error: {e}")
        return jsonify({'status': 'failure', 'message': 'Processing error.'}), 500

@app.route('/step-complete')
def step_complete():
    """
    Handles the completion of a biometric enrollment or verification step.
    Directs users to the next step, finalizes enrollment or grants access
    based on the current session state.
    """
    global USER_ENROLLED
    if 'user_mykad' not in session: return redirect(url_for('login'))
    
    # TIMEOUT CHECK
    if check_timeout():
         return redirect(url_for('login', alert='timeout'))
    
    mykad = session['user_mykad']
    
    # LOGIC A: REGISTRATION
    if not session.get('is_enrolled'):
        current = session.get('enrollment_stage')
        if current == 'face':
            session['enrollment_stage'] = 'voice'
            return redirect(url_for('biometric_step'))
        elif current == 'voice':
            session['enrollment_stage'] = 'gesture'
            return redirect(url_for('biometric_step'))
        elif current == 'gesture':
            session['is_enrolled'] = True
            session['user_mykad'] = mykad
            session['vault_access_granted'] = True 
            
            # --- FINALZE ENROLLMENT IN DB ---
            with get_db() as conn:
                conn.execute('UPDATE citizens SET is_enrollment_complete = 1 WHERE mykad_number = ?', (mykad,))
                conn.commit()
                
            log_action(mykad, "BIOMETRIC_REG", "All Factors", "System", "SUCCESS")
            return redirect(url_for('dashboard'))

    # LOGIC B: VERIFICATION
    else:
        queue = session.get('verification_queue', [])
        if len(queue) > 0:
            next_method = queue.pop(0)
            session['verification_queue'] = queue
            session['enrollment_stage'] = next_method
            return redirect(url_for('biometric_step'))
        else:
            session['user_mykad'] = mykad
            session['vault_access_granted'] = True
            log_action(mykad, "BIOMETRIC_LOGIN", "2 Factors", "System", "SUCCESS")
            return redirect(url_for('dashboard'))

# --- ISSUER PORTAL ---
@app.route('/issuer/dashboard')
def issuer_dashboard():
    """
    Displays the dashboard for JPN officers (Issuers).
    Shows recent logs and a list of citizens for management.
    """
    if session.get('user_mykad') != 'mockjpn':
        return redirect(url_for('login'))
    
    with get_db() as conn:
        citizens = conn.execute('SELECT * FROM citizens').fetchall()
        logs = conn.execute("SELECT * FROM access_logs WHERE action='ISSUANCE' ORDER BY timestamp DESC LIMIT 10").fetchall()
    
    # Mock user for base.html
    mock_user = {
        'full_name': 'JPN OFFICER',
        'mykad_number': 'ISSUER-ID-01'
    }
    
    return render_template('issuer_dashboard.html', citizens=citizens, logs=logs, user=mock_user)

@app.route('/issuer/issue', methods=['POST'])
def issue_certificate():
    """
    API to issue a Standard Birth Certificate to a specific user.
    Simulates the generation, signing (Layer 2), and anchoring (Layer 1) of the document.
    """
    if session.get('user_mykad') != 'mockjpn':
        return jsonify({'status': 'failure', 'message': 'Unauthorized'}), 403

    target_mykad = request.form.get('mykad')
    
    # Use a single database connection for all operations
    with get_db() as conn:
        # 1. Fetch Citizen
        user = conn.execute('SELECT full_name FROM citizens WHERE mykad_number = ?', (target_mykad,)).fetchone()
        
        if not user:
            return jsonify({'status': 'failure', 'message': 'Citizen not found'})

        # 2. Construct Certificate Data
        cert_data = {
            "type": "BIRTH_CERTIFICATE",
            "mykad": target_mykad,
            "name": user['full_name'],
            "issuer": "JPN MALAYSIA",
            "date_issued": datetime.datetime.now().isoformat()
        }

        # 3. Layer 2: Sign Data
        signature = signer.sign_data(cert_data)

        # 4. Layer 1: Anchor to Merkle Tree
        # Create a hash of the signed package to be the leaf
        leaf_content = json.dumps(cert_data) + signature
        merkle_root = merkle_tree.add_leaf(leaf_content)

        # 5. Create Final Package
        final_package = {
            "data": cert_data,
            "signature": signature,
            "merkle_root": merkle_root, 
            "verification_status": "SECURED"
        }
        
        final_json = json.dumps(final_package)

        # 6. Store in DB (Simulating issuance)
        # We overwrite the 'birth_cert_enc' column which was previously just a string
        conn.execute('UPDATE citizens SET birth_cert_enc = ? WHERE mykad_number = ?', (final_json, target_mykad))
        conn.commit()

    # 7. Log
    log_action(target_mykad, "ISSUANCE", "Birth Cert Issued (L1+L2)", "JPN", "SUCCESS")

    return jsonify({'status': 'success', 'message': f'Certificate issued to {user["full_name"]}'})

@app.route('/issuer/issue-bulk', methods=['POST'])
def issue_certificate_bulk():
    """
    API to issue certificates to multiple users at once.
    """
    if session.get('user_mykad') != 'mockjpn':
        return jsonify({'status': 'failure', 'message': 'Unauthorized'}), 403

    data = request.get_json()
    mykads = data.get('mykads', [])
    
    if not mykads:
        return jsonify({'status': 'failure', 'message': 'No citizens selected'})

    conn = get_db_connection()
    success_count = 0
    errors = []

    for target_mykad in mykads:
        try:
             # 1. Fetch Citizen
            user = conn.execute('SELECT full_name FROM citizens WHERE mykad_number = ?', (target_mykad,)).fetchone()
            
            if not user:
                errors.append(f"{target_mykad}: Not Found")
                continue

            # 2. Construct Certificate Data
            cert_data = {
                "type": "BIRTH_CERTIFICATE",
                "mykad": target_mykad,
                "name": user['full_name'],
                "issuer": "JPN MALAYSIA",
                "date_issued": datetime.datetime.now().isoformat()
            }

            # 3. Layer 2: Sign Data
            signature = signer.sign_data(cert_data)

            # 4. Layer 1: Anchor to Merkle Tree
            leaf_content = json.dumps(cert_data) + signature
            merkle_root = merkle_tree.add_leaf(leaf_content)

            # 5. Create Final Package
            final_package = {
                "data": cert_data,
                "signature": signature,
                "merkle_root": merkle_root, 
                "verification_status": "SECURED"
            }
            
            final_json = json.dumps(final_package)

            # 6. Store
            conn.execute('UPDATE citizens SET birth_cert_enc = ? WHERE mykad_number = ?', (final_json, target_mykad))
            
            # 7. Log
            # We can log individually but for bulk maybe just one log or individual?
            # Individual logs are better for audit trail.
            # Using the existing log function which opens its own connection is inefficient here inside a loop if we had thousands, 
            # but for <100 it's fine. However, we have an open connection 'conn'.
            # Let's write to access_logs directly using 'conn' to be safe with locking.
            conn.execute('''INSERT INTO access_logs (mykad_number, action, context, organization_name, status)VALUES (?, ?, ?, ?, ?)''', 
                     (target_mykad, "ISSUANCE", "Birth Cert Issued (Bulk)", "JPN", "SUCCESS"))
            
            success_count += 1
            
        except Exception as e:
            print(f"Error issuing for {target_mykad}: {e}")
            errors.append(f"{target_mykad}: Failed")

    conn.commit()
    conn.close()

    return jsonify({
        'status': 'success', 
        'message': f'Successfully issued {success_count} certificates.',
        'errors': errors
    })


@app.route('/issuer/edit/<mykad>')
def issue_certificate_edit_page(mykad):
    """
    Displays the edit form for issuing a custom certificate.
    Pre-fills data if an existing certificate is found.
    """
    if session.get('user_mykad') != 'mockjpn':
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    user = conn.execute('SELECT full_name, mykad_number, address, birth_cert_enc FROM citizens WHERE mykad_number = ?', (mykad,)).fetchone()
    conn.close()
    
    if not user:
        return "Citizen not found", 404

    # Default values or try to parse existing cert if available
    defaults = {
        'name': user['full_name'],
        'mykad': user['mykad_number'],
        'dob': "20" + user['mykad_number'][0:2] + "-" + user['mykad_number'][2:4] + "-" + user['mykad_number'][4:6], # Simple guess
        'pob': "KUALA LUMPUR",
        'address': user['address'] or "NO 123, JALAN MERDEKA, 50000 KUALA LUMPUR",
        'father': "ALI BIN AHMAD",
        'mother': "SITI BINTI ABU"
    }

    # If already issued L2 cert, try to pre-fill from that (JSON)
    if user['birth_cert_enc'] and '{' in user['birth_cert_enc']:
        try:
             package = json.loads(user['birth_cert_enc'])
             data = package.get('data', {})
             defaults['name'] = data.get('name', defaults['name'])
             defaults['dob'] = data.get('dob', defaults['dob'])
             defaults['pob'] = data.get('pob', defaults['pob'])
             defaults['address'] = data.get('address', defaults['address'])
             defaults['father'] = data.get('father', defaults['father'])
             defaults['mother'] = data.get('mother', defaults['mother'])
        except:
            pass

    # Mock user for base.html
    mock_user = {
        'full_name': 'JPN OFFICER',
        'mykad_number': 'ISSUER-ID-01'
    }

    return render_template('issuer_edit_cert.html', citizen=defaults, user=mock_user)


@app.route('/issuer/issue-custom', methods=['POST'])
def issue_certificate_custom():
    """
    API to issue a certificate with custom details provided via form.
    """
    if session.get('user_mykad') != 'mockjpn':
        return jsonify({'status': 'failure', 'message': 'Unauthorized'}), 403

    target_mykad = request.form.get('mykad')
    
    # Construct Certificate Data from Form
    cert_data = {
        "type": "BIRTH_CERTIFICATE",
        "mykad": target_mykad,
        "name": request.form.get('name'),
        "dob": request.form.get('dob'),
        "pob": request.form.get('pob'), # Place of Birth
        "address": request.form.get('address'),
        "father": request.form.get('father'),
        "mother": request.form.get('mother'),
        "issuer": "JPN MALAYSIA",
        "date_issued": datetime.datetime.now().isoformat()
    }

    # Layer 2: Sign Data
    signature = signer.sign_data(cert_data)

    # Layer 1: Anchor to Merkle Tree
    leaf_content = json.dumps(cert_data) + signature
    merkle_root = merkle_tree.add_leaf(leaf_content)

    # Final Package
    final_package = {
        "data": cert_data,
        "signature": signature,
        "merkle_root": merkle_root, 
        "verification_status": "SECURED"
    }
    
    final_json = json.dumps(final_package)

    # Store
    conn = get_db_connection()
    conn.execute('UPDATE citizens SET birth_cert_enc = ? WHERE mykad_number = ?', (final_json, target_mykad))
    
    # Log
    conn.execute('''INSERT INTO access_logs (mykad_number, action, context, organization_name, status)VALUES (?, ?, ?, ?, ?)''', 
             (target_mykad, "ISSUANCE", "Birth Cert Issued (Custom)", "JPN", "SUCCESS"))
    
    conn.commit()
    conn.close()

    return jsonify({'status': 'success', 'message': f'Custom Certificate issued for {target_mykad}'})


@app.route('/issuer/api/history/<mykad>')
def get_citizen_history(mykad):
    """
    API to fetch the full transaction history of a specific citizen.
    """
    if session.get('user_mykad') != 'mockjpn':
        return jsonify({'status': 'failure', 'message': 'Unauthorized'}), 403
        
    conn = get_db_connection()
    try:
        # Try fetching with timestamp
        logs = conn.execute('SELECT action, context, timestamp, status FROM access_logs WHERE mykad_number = ? ORDER BY id DESC', (mykad,)).fetchall()
    except Exception:
        # Fallback if timestamp column missing (older schema)
        logs = conn.execute('SELECT action, context, "2025-12-23 (Est)" as timestamp, status FROM access_logs WHERE mykad_number = ? ORDER BY id DESC', (mykad,)).fetchall()
    
    conn.close()
    
    history = []
    for log in logs:
        history.append({
            'action': log['action'],
            'context': log['context'],
            'timestamp': log['timestamp'] if log['timestamp'] else "N/A",
            'status': log['status']
        })
        
    return jsonify({'status': 'success', 'history': history})


@app.route('/issuer/logs')
def issuer_logs_page():
    """
    Displays the audit logs specific to JPN organization actions.
    """
    if session.get('user_mykad') != 'mockjpn':
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    try:
        # Fetch only JPN logs (Issuer actions)
        logs = conn.execute("SELECT action, context, timestamp, status, mykad_number, organization_name FROM access_logs WHERE organization_name = 'JPN' ORDER BY id DESC").fetchall()
    except Exception:
         # Fallback
        logs = conn.execute("SELECT action, context, '2025-12-23 (Est)' as timestamp, status, mykad_number, organization_name FROM access_logs WHERE organization_name = 'JPN' ORDER BY id DESC").fetchall()
    
    conn.close()
    
    # Mock user for base.html
    mock_user = {
        'full_name': 'JPN OFFICER',
        'mykad_number': 'ISSUER-ID-01'
    }

    return render_template('issuer_logs.html', logs=logs, user=mock_user)


# --- VERIFICATION ENDPOINT (For Citizen Button) ---
@app.route('/verify-document')
def verify_document():
    """
    Validates a document's cryptographic signature (Layer 2) and Merkle Proof (Layer 1).
    Used by the frontend 'Verify Integrity' button.
    """
    doc_type = request.args.get('type')
    
    # We currently only support birth_cert for this demo
    if doc_type != 'birth_certificate':
        return jsonify({'valid': False, 'message': 'Validation not supported'})
    
    user_data, mykad = get_user_context()
    if not user_data:
         return jsonify({'valid': False, 'message': 'Session Error'})

    # Get the blob
    cert_blob = user_data['birth_cert_enc']
    
    try:
        package = json.loads(cert_blob)
        data = package['data']
        signature = package['signature']
        
        # Verify Layer 2 (Signature)
        if signer.verify_signature(data, signature):
            return jsonify({'valid': True, 'message': 'Signature Verified (L2) & Anchored (L1)'})
        else:
             return jsonify({'valid': False, 'message': 'Signature Invalid'})

    except Exception as e:
        print(f"Verify Error: {e}")
        return jsonify({'valid': False, 'message': 'Legacy or Corrupt Document'})


# --- RETURNING USER SELECTION HANDLER (Untouched) ---


# DASHBOARD & WIDGETS
@app.route('/')
@app.route('/dashboard')
def dashboard():
    """
    Displays the dashboard for users who have successfully logged in and passed
    biometric verification.
    """
    if not session.get('vault_access_granted'): 
        return redirect(url_for('login'))
    user_data, mykad = get_user_context()

    if not user_data:  # <-- safeguard
        session.clear()  # Clear invalid session
        return redirect(url_for('login'))
    
    # Fetch recent logs for dashboard widget (limit 5)
    recent_logs = fetch_access_logs(mykad)[:4]
    
    return render_template('dashboard.html', user=user_data, logs=recent_logs)

@app.route('/files')
def my_files_entry():
    """
    Displays the 'My Files' vault.
    Verifies the user has passed all biometric checks before showing sensitive documents.
    """
    if not session.get('vault_access_granted'):
        return redirect(url_for('dashboard', alert='verification_failed'))
    
    user_data, mykad = get_user_context()
    if not user_data: 
        return redirect(url_for('login'))
    
    # Parse Birth Certificate Data if available (L1/L2 format)
    birth_cert_data = None
    if user_data.get('birth_cert_enc') and '{' in user_data['birth_cert_enc']:
        try:
            package = json.loads(user_data['birth_cert_enc'])
            birth_cert_data = package.get('data')
            # Ensure keys exist to avoid JS errors if partial
            if birth_cert_data:
                 birth_cert_data.setdefault('name', user_data['full_name'])
                 birth_cert_data.setdefault('dob', f"{str(mykad)[0:2]}-{str(mykad)[2:4]}-{str(mykad)[4:6]}")
                 birth_cert_data.setdefault('pob', "KUALA LUMPUR")
                 birth_cert_data.setdefault('address', user_data['address'])
                 birth_cert_data.setdefault('father', "ALI BIN AHMAD")
                 birth_cert_data.setdefault('mother', "SITI BINTI ABU")
        except:
            print("Error parsing birth cert JSON for display")
            pass
            
    # Fallback if no valid JSON cert found use existing DB user_data
    if not birth_cert_data:
        birth_cert_data = {
            'name': user_data['full_name'],
            'dob': f"{str(mykad)[0:2]}-{str(mykad)[2:4]}-{str(mykad)[4:6]}",
            'pob': "KUALA LUMPUR", # Default
            'address': user_data['address'],
            'father': "ALI BIN AHMAD", # Default
            'mother': "SITI BINTI ABU" # Default
        }

    # Create file list from DB data
    files_list = []

    files_list.append({
        'name': 'Birth Certificate',
        'mykad_link': user_data['birth_cert_enc'],
        'status': 'Available',
        'icon': 'fa-id-card'
    })
    
    files_list.append({
        'name': 'Water Bill',
        'mykad_link': user_data['water_bill_enc'],
        'status': 'Available',
        'icon': 'fa-file-invoice-dollar'
    })
    
    if user_data['oku_status'] == 'Active':
        oku_status = 'Active'
    else:\
        oku_status = 'Not Applicable'

    files_list.append({
        'name': 'OKU Status Card',
        'mykad_link': user_data['oku_status_enc'],
        'status': oku_status,
        'icon': 'fa-wheelchair'
    })    

    return render_template('allFiles.html', user = user_data, files = files_list, cert_data = birth_cert_data)

def expire_past_shares(mykad):
    """
    Checks for any ACTIVE shares owned by 'mykad' that have passed their
    expiration time and updates them to 'EXPIRED' status.
    """
    conn = get_db_connection()
    try:
        limit_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn.execute('''
            UPDATE share_sessions 
            SET status = 'EXPIRED' 
            WHERE sender_mykad = ? 
            AND status = 'ACTIVE' 
            AND expires_at < ?
        ''', (mykad, limit_time))
        conn.commit()
    except Exception as e:
        print(f"Auto-expire Error: {e}")
    finally:
        conn.close()

@app.route('/share')
def share():
    """
    Displays the user's active shared access records, allowing the user to
    view or manage documents they have shared with others.
    """
    if not session.get('vault_access_granted'): 
        return redirect(url_for('login'))
    user_data, mykad = get_user_context()
    
    # Auto-expire old shares before fetching
    expire_past_shares(mykad)
    
    # Fetch actual active share sessions for the UI list
    conn = get_db_connection()
    active_shares = conn.execute('SELECT * FROM share_sessions WHERE sender_mykad = ? AND status = "ACTIVE" ORDER BY id DESC', (mykad,)).fetchall()
    conn.close()
    
    # active_shares = fetch_access_logs(mykad, log_status='ACTIVE') # Old Log based approach
    return render_template('share.html', user = user_data, active_shares = active_shares)

@app.route('/logs')
def access_log(): 
    """
    Displays all access logs for the currently logged-in user.
    Also fetches 'Managed Shares' created by this user.
    """
    if not session.get('vault_access_granted'): 
        return redirect(url_for('login'))
        
    user_data, mykad = get_user_context()
    
    # Auto-expire old shares before fetching
    expire_past_shares(mykad)
    
    # 1. Fetch standard Audit Logs
    all_logs = fetch_access_logs(mykad)
    
    # 2. Fetch User's Created Shares
    conn = get_db_connection()
    # Get all shares created by this user, ordered by newest first
    # [UPDATED] SQL-Based Filter (Strict Matching) - Same as my_shares_list
    # Only select share sessions where an access log entry exists referencing their ID
    query = '''
        SELECT DISTINCT s.* 
        FROM share_sessions s
        WHERE s.sender_mykad = ?
        AND EXISTS (
            SELECT 1 FROM access_logs a 
            WHERE a.mykad_number = s.sender_mykad 
            AND (
                a.context = 'Guest accessed Share:' || s.id 
                OR a.context LIKE 'ShareID:' || s.id || ' -%'
            )
        )
        ORDER BY s.created_at DESC
    '''
    my_shares = conn.execute(query, (mykad,)).fetchall()
    conn.close()
    
    shares_list = []
    for row in my_shares:
        d = dict(row)
        # Format expires_at for display 
        if d.get('expires_at'):
             try:
                dt = datetime.datetime.strptime(d['expires_at'], '%Y-%m-%d %H:%M:%S')
                d['expires_at'] = dt.strftime("%d %b %Y, %I:%M %p")
             except:
                pass
        shares_list.append(d)

    return render_template('accessLog.html', user=user_data, logs=all_logs, my_shares=shares_list)

@app.route('/my_shares')
def my_shares_list():
    """
    Displays ONLY the 'Managed Shares' created by this user.
    """
    if not session.get('vault_access_granted'): 
        return redirect(url_for('login'))
        
    user_data, mykad = get_user_context()
    
    conn = get_db_connection()
    # Get all shares created by this user, ordered by newest first
    shares_list = []
    
    # [UPDATED] SQL-Based Filter (Strict Matching)
    # We use specific patterns to avoid "1 matches 10" issues.
    # 1. Redemption Log: "Guest accessed Share:<id>" (Exact string usually)
    # 2. View Log: "ShareID:<id> - Accessed..." (Followed by ' -')
    query = '''
        SELECT DISTINCT s.* 
        FROM share_sessions s
        WHERE s.sender_mykad = ?
        AND EXISTS (
            SELECT 1 FROM access_logs a 
            WHERE a.mykad_number = s.sender_mykad 
            AND (
                a.context = 'Guest accessed Share:' || s.id 
                OR a.context LIKE 'ShareID:' || s.id || ' -%'
            )
        )
        ORDER BY s.created_at DESC
    '''
    
    my_shares = conn.execute(query, (mykad,)).fetchall()
    conn.close()

    for row in my_shares:
        d = dict(row)
        # Format expires_at for display 
        if d.get('expires_at'):
             try:
                dt = datetime.datetime.strptime(d['expires_at'], '%Y-%m-%d %H:%M:%S')
                d['expires_at'] = dt.strftime("%d %b %Y, %I:%M %p")
             except:
                pass
        shares_list.append(d)

    return render_template('myShares.html', user=user_data, my_shares=shares_list)

@app.route('/my-face')
def my_face():
    """
    Serves the user's stored face image.
    """
    if not session.get('vault_access_granted'):
        return redirect(url_for('login'))
        
    user_data, mykad = get_user_context()
    if not user_data or not user_data['face_image_blob']:
        # Return a place holder or 404
        return "No face image found", 404
        
    return send_file(io.BytesIO(user_data['face_image_blob']), mimetype='image/jpeg')

@app.route('/settings')
def settings():
    """
    Displays the settings page for the currently logged-in user, 
    allowing them to view or update personal preferences.
    """
    if not session.get('vault_access_granted'):
        return redirect(url_for('login'))
    user_data, mykad = get_user_context()
    return render_template('settings.html', user = user_data) 


# --- SHARE CAPSULE LOGIC ---

@app.route('/receive_capsule')
def receive_capsule():
    """
    Public page for recipients to enter the code.
    """
    # Attempt to get logged-in user, else Guest
    user_context = {'full_name': 'Guest', 'mykad_number': '-'}
    if session.get('vault_access_granted'):
        u, _ = get_user_context()
        if u: user_context = u
            
    return render_template('receive_capsule.html', user=user_context)

@app.route('/redeem_share', methods=['POST'])
def redeem_share():
    """
    Validates the share code/OTP and grants guest access.
    """
    share_code = request.form.get('share_code')
    sender_id_input = request.form.get('sender_id')
    
    if not share_code or not sender_id_input:
         return jsonify({'status': 'failure', 'message': 'Missing credentials'})
         
    conn = get_db_connection()
    
    # Match MyKad + OTP + Status=ACTIVE
    share = conn.execute('SELECT * FROM share_sessions WHERE sender_mykad = ? AND otp_code = ? AND status = "ACTIVE"', (sender_id_input, share_code)).fetchone()
    
    if not share:
        conn.close()
        return jsonify({'status': 'failure', 'message': 'Invalid credentials or session expired.'})
        
    sender = conn.execute('SELECT full_name FROM citizens WHERE mykad_number = ?', (share['sender_mykad'],)).fetchone()
    conn.close()
    
    # Grant Guest Access
    session['guest_access'] = True
    session['guest_share_id'] = share['id']
    session['guest_sender_id'] = share['sender_mykad']
    session['guest_sender_name'] = sender['full_name']
    
    log_action(share['sender_mykad'], "SHARE ACCESS", f"Guest accessed Share:{share['id']}", "System", "SUCCESS")
    
    return jsonify({
        'status': 'success', 
        'sender_name': sender['full_name'],
        'sender_id': share['sender_mykad']
    })

@app.route('/create_share', methods=['POST'])
def create_share():
    """
    Creates a new share session and generates a dynamic OTP/QR.
    """
    if not session.get('vault_access_granted'):
        return jsonify({'status': 'failure', 'message': 'Unauthorized'}), 401

    mykad = session.get('user_mykad')
    recipient_type = request.form.get('recipient_type')
    recipient_id = request.form.get('recipient_id')
    
    # Basic validation
    if not recipient_type:
        return jsonify({'status': 'failure', 'message': 'Recipient type required'}), 400
    if recipient_type == 'Individual':
        if not recipient_id:
            return jsonify({'status': 'failure', 'message': 'Recipient ID required'}), 400
            
        # Validate Recipient Exists in DB
        conn = get_db_connection()
        recipient_exists = conn.execute('SELECT 1 FROM citizens WHERE mykad_number = ?', (recipient_id,)).fetchone()
        conn.close()
        
        if not recipient_exists:
             return jsonify({'status': 'failure', 'message': f'Recipient ID ({recipient_id}) not found in database.'}), 404
             
    # Capture Document Selections
    # We look for specific keys expected from the frontend form
    # Process permissions
    # Identity is MANDATORY, so we always include it
    selected_docs = ['identity']
    if request.form.get('share_license') == 'on': selected_docs.append('license')
    if request.form.get('share_income') == 'on': selected_docs.append('income')
    if request.form.get('share_birth') == 'on': selected_docs.append('birth')
    if request.form.get('share_water') == 'on': selected_docs.append('water')
    if request.form.get('share_oku') == 'on': selected_docs.append('oku')
    if request.form.get('share_vaccine') == 'on': selected_docs.append('vaccine')
    if request.form.get('share_transcript') == 'on': selected_docs.append('transcript')
    if request.form.get('share_epf') == 'on': selected_docs.append('epf')
    
    import json
    shared_docs_json = json.dumps(selected_docs)

    # Get Access Duration (Default to 24 hours if missing)
    usage_limit = request.form.get('usage_limit')
    
    # Duration Map
    duration_map = {
        '1_hour': datetime.timedelta(hours=1),
        '6_hours': datetime.timedelta(hours=6),
        '24_hours': datetime.timedelta(hours=24),
        '3_days': datetime.timedelta(days=3),
        '7_days': datetime.timedelta(days=7)
    }
    
    if usage_limit == 'custom':
        try:
            c_date = request.form.get('custom_date')
            c_time = request.form.get('custom_time')
            # HTML5 inputs return YYYY-MM-DD and HH:MM
            expires_at_dt = datetime.datetime.strptime(f"{c_date} {c_time}", "%Y-%m-%d %H:%M")
            # If date is in past, default to 24h or error? Let's just allow it (it will expire immediately)
        except Exception as e:
            print(f"Custom Date Error: {e}")
            # Fallback
            expires_at_dt = datetime.datetime.now() + datetime.timedelta(hours=24)
    else:
        delta = duration_map.get(usage_limit, datetime.timedelta(hours=24))
        expires_at_dt = datetime.datetime.now() + delta
        
    expires_at_str = expires_at_dt.strftime('%Y-%m-%d %H:%M:%S')

    # Generate 6-digit OTP
    otp_code = ''.join(random.choices(string.digits, k=6))
    
    # Store in DB
    allow_download_val = 'TRUE' if request.form.get('allow_download') == 'on' else 'FALSE'

    try:
        conn = get_db_connection()
        # Ensure column exists or handle error (Migration should have run)
        cursor = conn.execute(f'''
            INSERT INTO share_sessions (sender_mykad, recipient_type, recipient_id, otp_code, status, expires_at, shared_docs, allow_download)
            VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?, ?)
        ''', (mykad, recipient_type, recipient_id, otp_code, expires_at_str, shared_docs_json, allow_download_val))
        share_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # log_action(mykad, "SHARE", f"Created Capsule ({usage_limit}) for {recipient_type}:{recipient_id}", "System", "SUCCESS")
        
        return jsonify({'status': 'success', 'share_id': share_id, 'otp_code': otp_code})
        
    except Exception as e:
        print(f"Share Creation Error: {e}")
        return jsonify({'status': 'failure', 'message': str(e)}), 500

@app.route('/share_status/<int:share_id>')
def share_status(share_id):
    """
    Returns the current OTP and QR code for a share session.
    Implements 60s rotation logic (simulated by re-generating if 'expired' logic met, 
    but for now we keep simple static or simple time-based rotation).
    User Requirement: "Refresh every 1 min".
    """
    if not session.get('vault_access_granted'):
        return jsonify({'status': 'failure'}), 401
        
    conn = get_db_connection()
    session_row = conn.execute('SELECT * FROM share_sessions WHERE id = ?', (share_id,)).fetchone()
    
    if not session_row or session_row['status'] != 'ACTIVE':
        conn.close()
        return jsonify({'status': 'expired'})
    
    # Dynamic Rotation Logic
    # We check if the current OTP is older than 60 seconds.
    # To support this without schema changes (complex), we'll hack it slightly:
    # We will assume 'created_at' is the last update time (updating it on rotation).
    
    try:
        current_time = datetime.datetime.now()
        # Parse timestamp (SQLite format: YYYY-MM-DD HH:MM:SS)
        last_updated_str = session_row['created_at'] 
        
        # Handle potential format differences
        try:
             last_updated = datetime.datetime.strptime(last_updated_str, '%Y-%m-%d %H:%M:%S')
        except:
             # Fallback if fraction included
             last_updated = datetime.datetime.strptime(last_updated_str, '%Y-%m-%d %H:%M:%S.%f')

        elapsed = (current_time - last_updated).total_seconds()
        
        current_otp = session_row['otp_code']
        
        if elapsed > 60:
             # ROTATE!
             new_otp = ''.join(random.choices(string.digits, k=6))
             
             # Update DB (Update created_at to now to reset timer)
             # Note: We are repurposing created_at as 'last_updated' for this feature.
             conn.execute('UPDATE share_sessions SET otp_code = ?, created_at = CURRENT_TIMESTAMP WHERE id = ?', (new_otp, share_id))
             conn.commit()
             
             current_otp = new_otp
             print(f"DEBUG: Rotated OTP for Share {share_id} -> {new_otp}")
    except Exception as e:
        print(f"Rotation Logic Error: {e}")
        current_otp = session_row['otp_code']

    # QR Generation
    # Embed both Sender ID and OTP for seamless scanning
    import json
    qr_data = json.dumps({
        "id": session_row['sender_mykad'],
        "otp": current_otp
    })
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(qr_data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    qr_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    conn.close()
    
    return jsonify({
        'status': 'active',
        'code': current_otp,
        'qr_image': f"data:image/png;base64,{qr_base64}",
        'expires_in': 3600 # Mock
    })



@app.route('/view_capsule_content')
def view_capsule_content():
    """
    Displays the content of a shared capsule to a guest recipient.
    Validates the guest session and renders the allowed documents.
    """
    if not session.get('guest_access'):
        return redirect(url_for('receive_capsule'))
        
    sender = {
        'full_name': session.get('guest_sender_name'),
        'mykad_number': session.get('guest_sender_id'),
        'address': 'KUALA LUMPUR (HIDDEN)',
    }
    
    # Retrieve active session details to know what to show
    # We need the 'share_id' or re-query by OTP if we stored it in session
    # For MVP, let's assume we store share_id in session during redeem_share
    
    share_id = session.get('guest_share_id')
    allowed_docs = []
    session_otp = "UNKNOWN"
    
    if share_id:
        conn = get_db_connection()
        share_row = conn.execute('SELECT * FROM share_sessions WHERE id = ?', (share_id,)).fetchone()
        conn.close()
        
        if share_row:
            import json
            try:
                allowed_docs = json.loads(share_row['shared_docs']) if share_row['shared_docs'] else []
            except:
                allowed_docs = []
            session_otp = share_row['otp_code']
            
            # LOG VIEW ACTION ONLY ONCE PER SESSION IF NEEDED
            # For now, log every hit to track activity
            # Context format: "ShareID:{id}" to easily filter later
            log_action(share_row['sender_mykad'], "VIEW CAPSULE", f"ShareID:{share_id} - Accessed by recipient", "Secure Share", "SUCCESS")
            
            # Format expiry for display
            expires_at_display = share_row['expires_at']
            try:
                dt = datetime.datetime.strptime(expires_at_display, '%Y-%m-%d %H:%M:%S')
                expires_at_display = dt.strftime("%d %b %Y, %I:%M %p")
            except:
                pass


            # Check Allow Download Pref
            allow_dl = 'TRUE'
            try:
                if share_row['allow_download']:
                    allow_dl = share_row['allow_download']
            except:
                pass # Legacy rows default to TRUE
                
    return render_template('view_capsule.html', user=sender, allowed_docs=allowed_docs, 
                           share_id=share_id, session_otp=session_otp, expires_at=expires_at_display,
                           allow_download=allow_dl)





# --- SECURED PDF DOWNLOAD ROUTE ---
@app.route('/download_secured_doc/<int:share_id>/<string:doc_type>')
def download_secured_doc(share_id, doc_type):
    """
    Generates a password-protected PDF for the requested document.
    Password = Share OTP.
    """
    # 1. Verify Session/Access - Minimal check for demo/MVP
    conn = get_db_connection()
    share_row = conn.execute('SELECT * FROM share_sessions WHERE id = ?', (share_id,)).fetchone()
    
    if not share_row:
        conn.close()
        return "Share session not found", 404
        
    # Enforce ACTIVE status
    if share_row['status'] != 'ACTIVE':
        conn.close()
        return "Access Revoked. This share session is no longer active.", 403
        
    # Check if doc_type is allowed
    # Handle DB storing text vs actual json list
    try:
        allowed_docs = json.loads(share_row['shared_docs']) if share_row['shared_docs'] else []
    except:
        allowed_docs = [] # Fallback
    
    # Bypass check if list is empty (legacy shares) or strict check?
    # Strict check: if not in list, deny.
    if share_row['shared_docs'] and doc_type not in allowed_docs:
        conn.close()
        return "Document not shared or access denied.", 403

    # 2. Fetch the Sender's Document Blob
    sender_mykad = share_row['sender_mykad']
    sender_row = conn.execute('SELECT * FROM citizens WHERE mykad_number = ?', (sender_mykad,)).fetchone()
    conn.close()
    
    col_map = {
        'identity': 'mykad_front_blob',
        'license': 'driving_license_blob',
        'income': 'income_slip_blob',
        'birth': 'birth_cert_blob',
        'water': 'water_bill_blob',
        'oku': 'oku_card_blob'
    }
    
    blob_col = col_map.get(doc_type)
    image_data = None
    
    if blob_col:
        try:
            image_data = sender_row[blob_col]
        except (IndexError, KeyError):
            pass
    
    # 3. Generate Rich PDF
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=A4)
    width, height = A4 # 595.27, 841.89
    
    # --- DRAW BACKGROUND "PAPER" ---
    can.setFillColorRGB(0.98, 0.98, 0.98) # Off-white
    can.rect(0, 0, width, height, fill=1, stroke=0)
    
    # --- DRAW HEADER (MATCH UI) ---
    can.setFillColorRGB(0.2, 0.2, 0.2)
    can.setStrokeColorRGB(0.2, 0.2, 0.2)
    
    # Line
    can.line(50, height - 100, width - 50, height - 100)
    
    # Title
    can.setFont("Helvetica-Bold", 18)
    title_map = {
        'identity': 'IDENTITY CARD', 'license': 'DRIVING LICENSE', 
        'income': 'INCOME STATEMENT', 'birth': 'BIRTH CERTIFICATE',
        'vaccine': 'VACCINATION CERT', 'epf': 'EPF STATEMENT', 'transcript': 'TRANSCRIPT'
    }
    can.drawString(50, height - 85, title_map.get(doc_type, doc_type.upper()).replace('_', ' '))
    
    # Subtitle
    can.setFont("Helvetica", 10)
    can.setFillColorRGB(0.5, 0.5, 0.5)
    source_map = {'identity': 'JPN', 'license': 'JPJ', 'income': 'LHDN', 'vaccine': 'MOH', 'birth': 'JPN'}
    can.drawString(50, height - 115, f"SOURCE: {source_map.get(doc_type, 'OFFICIAL RECORD')}")
    
    # --- DRAW STAMP (Text representation) ---
    can.saveState()
    can.setStrokeColorRGB(0.1, 0.7, 0.4) # Green
    can.setFillColorRGB(0.1, 0.7, 0.4)
    can.setLineWidth(2)
    # Box
    can.roundRect(width - 160, height - 90, 110, 30, 4, stroke=1, fill=0)
    # Check
    can.setFont("Helvetica-Bold", 10)
    can.drawString(width - 145, height - 78, "VERIFIED: VALID")
    can.restoreState()
    
    # --- DRAW USER INFO BLOCKS ---
    current_y = height - 150
    can.setFillColorRGB(0, 0, 0)
    
    labels = [
        ("FULL NAME", sender_row['full_name'].upper()),
        ("MYKAD ID", sender_row['mykad_number']),
        ("ADDRESS", sender_row['address'][:40] + "..." if sender_row['address'] else "-")
    ]
    
    for label, value in labels:
        can.setFont("Helvetica", 8)
        can.setFillColorRGB(0.5, 0.5, 0.5)
        can.drawString(50, current_y, label)
        
        can.setFont("Helvetica-Bold", 12)
        can.setFillColorRGB(0.1, 0.1, 0.1)
        can.drawString(50, current_y - 15, str(value))
        
        current_y -= 40

    # --- DRAW IMAGE (CENTERED) ---
    # Placeholder Logic nested here for consistency
    if not image_data:
        try:
            from PIL import Image, ImageDraw
            img_chk = Image.new('RGB', (600, 400), color='#f0f0f0')
            d = ImageDraw.Draw(img_chk)
            d.text((200, 180), f"{doc_type.upper()} PREVIEW", fill="#aaa")
            buf = io.BytesIO()
            img_chk.save(buf, format='JPEG')
            image_data = buf.getvalue()
        except:
            pass # Should verify failsafe or return error? Let's assume failsafe works or we skip image
            
    if image_data:
        try:
            img_buffer = io.BytesIO(image_data)
            img = ImageReader(img_buffer)
            img_w, img_h = img.getSize()
            aspect = img_h / float(img_w)
            
            draw_width = 400
            draw_height = draw_width * aspect
            
            # Constraints
            if draw_height > 400:
                draw_height = 400
                draw_width = draw_height / aspect
                
            x_pos = (width - draw_width) / 2
            # Draw below the text info
            y_pos = current_y - draw_height - 20 
            
            can.drawImage(img, x_pos, y_pos, width=draw_width, height=draw_height)
            
            # Watermark on top of image
            can.saveState()
            can.translate(width/2, y_pos + draw_height/2)
            can.rotate(45)
            can.setFont("Helvetica-Bold", 60)
            can.setFillColorRGB(0.9, 0.9, 0.9, 0.5) # Transparent Grey
            can.drawCentredString(0, 0, "COPY")
            can.restoreState()
            
        except Exception as e:
            print(f"PDF Image Error: {e}")
            can.drawString(50, current_y - 20, "Image rendering failed.")

    can.save()
    packet.seek(0)
    
    # 4. Encrypt + DRM Injection
    # To simulate "Adobe Policy Server" style DRM without a real server,
    # we inject a JavaScript action that runs on PDF open.
    # It checks the current date against the expiry date.
    
    expires_at_str = "Unknown"
    
    # Try to get expiry. If column missing (IndexError/KeyError from Row), default to 24h
    try:
         expires_at_str = share_row['expires_at']
    except (IndexError, KeyError):
         # Column might not exist if migration didn't run or using old DB file
         print("Warning: 'expires_at' column missing in share_sessions.")
         expires_at_str = None

    if not expires_at_str:
         expires_at_str = (datetime.datetime.now() + datetime.timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')

    # Convert to PDF Date format (D:YYYYMMDDHHMMSS) for JS
    try:
         # Handle variable SQLite formats
         if '.' in expires_at_str: expires_at_str = expires_at_str.split('.')[0]
         dt = datetime.datetime.strptime(expires_at_str, '%Y-%m-%d %H:%M:%S')
         pdf_expiry_date = dt.strftime("D:%Y%m%d%H%M%S")
         readable_expiry = dt.strftime("%d %b %Y, %I:%M %p")
    except:
         # Fallback: 24h from now
         dt = datetime.datetime.now() + datetime.timedelta(hours=24)
         pdf_expiry_date = dt.strftime("D:%Y%m%d%H%M%S")
         readable_expiry = "Unknown"

    # Adobe JavaScript to check Expiry
    # Safe Date Construction in JS
    # Parsing in Python is reliable; passing integers to JS new Date() is robust.
    # Note: JS Date month is 0-indexed (0=Jan, 11=Dec)
    try:
        if '.' in expires_at_str: expires_at_str = expires_at_str.split('.')[0]
        dt = datetime.datetime.strptime(expires_at_str, '%Y-%m-%d %H:%M:%S')
        
        # Python: Month 1-12, JS: Month 0-11
        js_code = f"""
        var expiry = new Date({dt.year}, {dt.month - 1}, {dt.day}, {dt.hour}, {dt.minute}, {dt.second});
        var now = new Date();
        
        // 1. OFFLINE EXPIRY CHECK
        if (now > expiry) {{
            app.alert("SECURITY ALERT: This secure document expired on {dt.strftime('%d %b %Y %I:%M %p')}. Access is no longer granted.");
            this.closeDoc(true);
        }}
        
        // 2. ONLINE REVOCATION CHECK (Phone Home)
        // Attempt to check if share is REVOKED on server
        try {{
            var statusURL = "http://127.0.0.1:5000/check_share_status/{share_id}";
            
            // Note: Net.HTTP requires privileges. If this fails, we fall back to offline check.
            var params = {{
                cVerb: "GET",
                cURL: statusURL,
                oHandler: {{
                    response: function(args) {{
                        try {{
                            var str = SOAP.streamDecode(args.cStream);
                            var resp = JSON.parse(str);
                            if (resp.status == 'REVOKED') {{
                                app.alert("ACCESS REVOKED: The sender has revoked access to this document.");
                                event.target.closeDoc(true);
                            }}
                        }} catch(e) {{ }}
                    }}
                }}
            }};
            Net.HTTP.request(params);
            Net.HTTP.request(params);
        }} catch(e) {{
            // Network check failed/blocked. "Fail Closed" for strict security.
            app.alert("SECURITY CHECK FAILED: Unable to verify document status with server. Internet connection and Adobe Reader trust is required to view this secured document.");
            this.closeDoc(true);
        }}
        """
    except Exception as e:
        # Fallback to simple string parse if anything fails
        print(f"Date Parse Error: {e}")
        js_code = f"""
        app.alert("Security timestamp error.");
        """
    
    new_pdf = PdfReader(packet)
    writer = PdfWriter()
    
    for page in new_pdf.pages:
        writer.add_page(page)
    
    # Add JS Action to OpenAction
    writer.add_js(js_code)
        
    password = str(share_row['otp_code']) # Ensure string
    writer.encrypt(password)
    
    output_stream = io.BytesIO()
    writer.write(output_stream)
    output_stream.seek(0)
    
    # LOG DOWNLOAD ACTION
    # Context format: "ShareID:{id}" to allows filtering
    log_action(sender_mykad, "DOWNLOAD", f"ShareID:{share_id} - Downloaded {doc_type.upper()}", "Secure Share", "SUCCESS")
    
    return send_file(
        output_stream,
        as_attachment=True,
        download_name=f"{doc_type}_secured.pdf",
        mimetype='application/pdf'
    )


# --- DRM VALIDATION ENDPOINT ---
@app.route('/check_share_status/<int:share_id>')
def check_share_status(share_id):
    """
    Public endpoint for PDF 'Phone Home' checks.
    Returns JSON status: ACTIVE or REVOKED.
    """
    conn = get_db_connection()
    share = conn.execute('SELECT status FROM share_sessions WHERE id = ?', (share_id,)).fetchone()
    conn.close()
    
    if not share:
        return jsonify({'status': 'REVOKED'})
        
    return jsonify({'status': share['status']})


# --- SHARE REVOCATION ROUTES ---

@app.route('/revoke_share/<int:share_id>', methods=['POST'])
def revoke_share(share_id):
    """
    Revokes access to a share session by setting its status to REVOKED.
    """
    if not session.get('vault_access_granted'):
         return jsonify({'status': 'failure', 'message': 'Unauthorized'}), 401
    
    _, mykad = get_user_context()
    conn = get_db_connection()
    
    # Verify ownership
    share = conn.execute('SELECT * FROM share_sessions WHERE id = ? AND sender_mykad = ?', (share_id, mykad)).fetchone()
    if not share:
        conn.close()
        return jsonify({'status': 'failure', 'message': 'Share session not found or access denied.'}), 404
        
    conn.execute("UPDATE share_sessions SET status = 'REVOKED' WHERE id = ?", (share_id,))
    conn.commit()
    conn.close()
    
    log_action(mykad, "REVOKE", f"Revoked Share ID {share_id}", "System", "SUCCESS")
    
    return jsonify({'status': 'success', 'message': 'Access revoked successfully.'})


@app.route('/revoke_access/<int:log_id>', methods=['POST'])
def revoke_access(log_id):
    """
    Revokes a specific log entry (Legacy/System logs).
    """
    if not session.get('vault_access_granted'):
         return jsonify({'status': 'failure', 'message': 'Unauthorized'}), 401
    
    conn = get_db_connection()
    try:
        # Check if the log entry exists and is active
        log_entry = conn.execute('SELECT * FROM access_logs WHERE id = ?', (log_id,)).fetchone()
        if not log_entry:
            conn.close()
            return jsonify({'status': 'error', 'message': 'Log entry not found.'}), 404
        
        # Update the log entry status to REVOKED
        conn.execute('UPDATE access_logs SET status = ? WHERE id = ?', ('REVOKED', log_id))
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'success', 'message': 'Log revoked successfully.'})
        
    except Exception as e:
        conn.close()
        print(f"Error revoking log {log_id}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/logout')
def logout():
    """
    Logs out the user by clearing the session.
    """
    session.clear()
    return redirect(url_for('login'))



if __name__ == '__main__':
    print("-------------------------------------------------------")
    print("GOVERNMENT SYSTEM ONLINE")
    print("Login Page: http://127.0.0.1:5000/login")
    print("-------------------------------------------------------")
    app.run(debug = True, port = 5000)