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
import numpy as np

import numpy as np
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

# --- MEDIAPIPE HELPERS ---
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
        
        # Flatten: [x1, y1, z1, x2, y2, z2, ...]
        vector = []
        for lm in landmarks:
            vector.extend([lm.x, lm.y, lm.z])
            
        return np.array(vector)
        
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
        
    # Euclidean Distance
    dist = np.linalg.norm(v1 - v2)
    return dist
# --- END MEDIAPIPE HELPERS ---

def fetch_user_record_by_mykad(mykad):
    """
    Fetches a citizen's record using their MyKad number.
    """
    try:
        conn = get_db_connection()
        user_record = conn.execute('SELECT * FROM citizens WHERE mykad_number = ?', (mykad,)).fetchone()
        conn.close()
        return user_record
    except sqlite3.OperationalError as e:
        print(f"DATABASE ERROR: {e}")
        return None 

def log_action(mykad, action, context, organization_name, status = "SUCCESS"):
    """
    Inserts a new audit log entry.
    """
    try:
        conn = get_db_connection()
        conn.execute('''INSERT INTO access_logs (mykad_number, action, context, organization_name, status)VALUES (?, ?, ?, ?, ?)''', 
                     (mykad, action, context, organization_name, status))
        conn.commit()
        conn.close()
    except sqlite3.OperationalError as e:
        print(f"LOGGING ERROR: Could not write to access_logs. Did you run seed.py? Error: {e}")

def fetch_access_logs(mykad, log_status = None):
    """
    Fetches audit logs for a specific user, optionally filtered by status.
    """
    try:
        conn = get_db_connection()
        query = 'SELECT * FROM access_logs WHERE mykad_number = ? ORDER BY timestamp DESC'
        params = (mykad,)
        if log_status:
            query = 'SELECT * FROM access_logs WHERE mykad_number = ? AND status = ? ORDER BY timestamp DESC'
            params = (mykad, log_status)
        logs = conn.execute(query, params).fetchall()
        conn.close()
        readable_logs = []
        for log in logs:
            readable_logs.append(dict(log))
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

    conn = get_db_connection()
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
            conn.execute('UPDATE citizens SET failed_attempts = ?, lockout_until = ? WHERE mykad_number = ?', 
                         (current_attempts, lockout_until, mykad))
            print(f"DEBUG: {mykad} LOCKED for {lock_duration} mins")
            log_action(mykad, "LOGIN", f"Blocked (Locked for {lock_duration}m due to {reason})", "System", "BLOCKED")
        else:
            conn.execute('UPDATE citizens SET failed_attempts = ? WHERE mykad_number = ?', 
                         (current_attempts, mykad))
            log_action(mykad, "LOGIN", f"{reason} Attempt ({current_attempts})", "System", "FAILED")
            
        conn.commit()
    conn.close()

@app.route('/handle_login', methods = ['POST'])
def handle_login():
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
            conn = get_db_connection()
            conn.execute('UPDATE citizens SET failed_attempts = 0, lockout_until = NULL WHERE mykad_number = ?', (mykad,))
            conn.commit()
            conn.close()

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
            # Store in session to avoid global variable issues
            # Require BOTH Face and Voice to be considered "fully enrolled"
            if user_record['voice_audio_blob'] and user_record['face_encoding_blob']:
                session['is_enrolled'] = True
            else:
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
            conn = get_db_connection()
            # Store BOTH the image and the encoding
            conn.execute('UPDATE citizens SET face_image_blob = ?, face_encoding_blob = ? WHERE mykad_number = ?', 
                         (image_blob, new_encoding_blob, mykad))
            conn.commit()
            conn.close()
            
            log_action(mykad, "ENROLL", "Face Biometric Registered (MP)", "System", "SUCCESS")
            return jsonify({'status': 'success', 'message': 'Face registered successfully.'})

        # 7. VERIFICATION (Returning)
        else:
            conn = get_db_connection()
            # Fetch stored encoding from citizens table
            user_record = conn.execute('SELECT face_encoding_blob FROM citizens WHERE mykad_number = ?', (mykad,)).fetchone()
            
            if not user_record or not user_record['face_encoding_blob']:
                conn.close()
                return jsonify({'status': 'failure', 'message': 'No face enrolled. Please enroll first.'})

            stored_vector = pickle.loads(user_record['face_encoding_blob'])
            
            # Compare
            distance = compare_vectors(stored_vector, face_vector)
            print(f"DEBUG: Face Distance = {distance:.4f}")
            
            # Threshold: < 3.5 roughly match for same person/pose
            if distance < 3.5:
                conn.close()
                log_action(mykad, "VERIFY", "Face Match Success", "System", "SUCCESS")
                return jsonify({'status': 'success', 'message': 'Face verified successfully.'})
            else:
                conn.close()
                #print(f"DEBUG: Mismatch. Dist={distance}")
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
            conn = get_db_connection()
            conn.execute('UPDATE citizens SET voice_audio_blob = ? WHERE mykad_number = ?', (audio_data_blob, mykad))
            conn.commit()
            print(f"DEBUG: Voice BLOB saved.")
            conn.close()
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
        conn = get_db_connection()
        user = conn.execute('SELECT voice_audio_blob FROM citizens WHERE mykad_number = ?', (mykad,)).fetchone()
        conn.close()
        
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
            
            # 4. Threshold Check
            # Typical threshold for high security is around 0.75 - 0.80
            BIOMETRIC_THRESHOLD = 0.85 
            
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
        conn = get_db_connection()
        user_record = conn.execute('SELECT face_encoding_blob FROM citizens WHERE mykad_number = ?', (mykad,)).fetchone()
        conn.close()

        if not user_record or not user_record['face_encoding_blob']:
             return jsonify({'status': 'failure', 'message': 'No enrolled face data found.'})

        stored_vector = pickle.loads(user_record['face_encoding_blob'])
        
        # Compare
        distance = compare_vectors(stored_vector, new_vector)
        print(f"DEBUG: Gesture Face Dist = {distance:.4f}")
        
        # Threshold similar to verify_face
        if distance > 4.0: # Slightly looser for gestures
             log_action(mykad, "GESTURE_VERIFY", f"Face Mismatch (Dist={distance:.2f})", "System", "FAILED")
             return jsonify({'status': 'failure', 'message': 'Identity verification failed. Face does not match.'})

        # 4. Identity Confirmed
        print(f"DEBUG: Gesture Identity Verified. Dist={distance:.2f}")
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
    if session.get('user_mykad') != 'mockjpn':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    citizens = conn.execute('SELECT * FROM citizens').fetchall()
    logs = conn.execute("SELECT * FROM access_logs WHERE action='ISSUANCE' ORDER BY timestamp DESC LIMIT 10").fetchall()
    conn.close()
    
    # Mock user for base.html
    mock_user = {
        'full_name': 'JPN OFFICER',
        'mykad_number': 'ISSUER-ID-01'
    }
    
    return render_template('issuer_dashboard.html', citizens=citizens, logs=logs, user=mock_user)

@app.route('/issuer/issue', methods=['POST'])
def issue_certificate():
    if session.get('user_mykad') != 'mockjpn':
        return jsonify({'status': 'failure', 'message': 'Unauthorized'}), 403

    target_mykad = request.form.get('mykad')
    
    # 1. Fetch Citizen
    conn = get_db_connection()
    user = conn.execute('SELECT full_name FROM citizens WHERE mykad_number = ?', (target_mykad,)).fetchone()
    
    if not user:
        conn.close()
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
    conn.close()

    # 7. Log
    log_action(target_mykad, "ISSUANCE", "Birth Cert Issued (L1+L2)", "JPN", "SUCCESS")

    return jsonify({'status': 'success', 'message': f'Certificate issued to {user["full_name"]}'})

@app.route('/issuer/issue-bulk', methods=['POST'])
def issue_certificate_bulk():
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

@app.route('/share')
def share():
    """
    Displays the user's active shared access records, allowing the user to
    view or manage documents they have shared with others.
    """
    if not session.get('vault_access_granted'): 
        return redirect(url_for('login'))
    user_data, mykad = get_user_context()
    
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
    
    # 1. Fetch standard Audit Logs
    all_logs = fetch_access_logs(mykad)
    
    # 2. Fetch User's Created Shares
    conn = get_db_connection()
    # Get all shares created by this user, ordered by newest first
    my_shares = conn.execute('SELECT * FROM share_sessions WHERE sender_mykad = ? ORDER BY created_at DESC', (mykad,)).fetchall()
    conn.close()
    
    shares_list = [dict(row) for row in my_shares]

    return render_template('accessLog.html', user=user_data, logs=all_logs, my_shares=shares_list)

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

@app.route('/revoke_access/<int:log_id>', methods = ['POST'])
def revoke_access(log_id):
    """
    Revokes a previously shared access entry for the logged-in user.
    """
    if not session.get('vault_access_granted'): return jsonify({'status': 'failure', 'message': 'Unauthorized'}), 401
    mykad = session.get('user_mykad')
    conn = get_db_connection()
    try:
        cursor = conn.execute("UPDATE access_logs SET status = 'REVOKED' WHERE id = ? AND mykad_number = ? AND status = 'ACTIVE'", (log_id, mykad))
        if cursor.rowcount == 0:
            conn.close()
            return jsonify({'status': 'failure', 'message': 'Access not found.'}), 404
        conn.commit()
        conn.close()
        log_action(mykad, "REVOKE", f"Log ID: {log_id}", "User", "SUCCESS")
        return jsonify({'status': 'success', 'message': 'Revoked!'})
    except Exception as e:
        return jsonify({'status': 'failure', 'message': str(e)}), 500

# --- SHARE CAPSULE LOGIC ---

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
    selected_docs = []
    # Currently supported/hardcoded keys in share.html
    if request.form.get('share_identity') == 'on': selected_docs.append('identity')
    if request.form.get('share_license') == 'on': selected_docs.append('license')
    if request.form.get('share_income') == 'on': selected_docs.append('income')
    if request.form.get('share_birth') == 'on': selected_docs.append('birth')
    if request.form.get('share_water') == 'on': selected_docs.append('water')
    if request.form.get('share_oku') == 'on': selected_docs.append('oku')
    
    import json
    shared_docs_json = json.dumps(selected_docs)

    # Generate 6-digit OTP
    otp_code = ''.join(random.choices(string.digits, k=6))
    
    # Store in DB
    try:
        conn = get_db_connection()
        # Ensure column exists or handle error (Migration should have run)
        cursor = conn.execute('''
            INSERT INTO share_sessions (sender_mykad, recipient_type, recipient_id, otp_code, status, expires_at, shared_docs)
            VALUES (?, ?, ?, ?, 'ACTIVE', datetime('now', '+1 hour'), ?)
        ''', (mykad, recipient_type, recipient_id, otp_code, shared_docs_json))
        share_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        log_action(mykad, "SHARE", f"Created Capsule for {recipient_type}:{recipient_id}", "System", "SUCCESS")
        
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
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(current_otp)
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
    Validates the 6-digit code and shows the documents.
    """
    share_code = request.form.get('share_code')
    input_sender_id = request.form.get('sender_id') # New input
    
    conn = get_db_connection()
    # Find session by OTP Code AND status ACTIVE
    session_row = conn.execute('SELECT * FROM share_sessions WHERE otp_code = ? AND status = ?', (share_code, 'ACTIVE')).fetchone()
    
    if not session_row:
        conn.close()
        return jsonify({'status': 'error', 'message': 'Invalid or expired code.'})
        
    # Validation: Check if input_sender_id matches actual sender
    actual_sender_id = session_row['sender_mykad']
    if input_sender_id and input_sender_id.strip() != actual_sender_id:
         conn.close()
         return jsonify({'status': 'error', 'message': 'Sender ID does not match the code provided.'})

    # Verify Sender Exists in DB (Data Integrity Check)
    sender = conn.execute('SELECT full_name, mykad_number FROM citizens WHERE mykad_number = ?', (actual_sender_id,)).fetchone()
    if not sender:
        conn.close()
        return jsonify({'status': 'error', 'message': 'Sender account no longer exists.'})
    conn.close()
    
    # Return success and maybe the preview HTML directly or JSON payload
    # In a real flow, we would set a session token for the guest and redirect them.
    # For this demo, we can grant a temporary session variable.
    session['guest_access'] = True
    session['guest_sender_name'] = sender['full_name']
    session['guest_sender_id'] = sender['mykad_number']
    session['guest_share_id'] = session_row['id'] # Store ID for retrieval
    
    return jsonify({
        'status': 'success',
        'sender_name': sender['full_name'],
        'sender_id': sender['mykad_number']
    })

@app.route('/view_capsule_content')
def view_capsule_content():
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

    return render_template('view_capsule.html', user=sender, allowed_docs=allowed_docs, session_otp=session_otp, share_id=share_id)


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    """
    Logs out the current user by clearing all session data.
    """
    session.clear()
    return redirect(url_for('login'))


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
    if not blob_col or not sender_row[blob_col]:
        return "Document source file not found.", 404

    image_data = sender_row[blob_col]
    
    # 3. Generate PDF
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=A4)
    width, height = A4
    
    # Header
    can.setFont("Helvetica-Bold", 16)
    can.drawString(50, height - 50, f"Secured Document: {doc_type.upper()}")
    can.setFont("Helvetica", 10)
    can.drawString(50, height - 70, "Provide Share Code (OTP) to open.")
    
    # Image
    try:
        img_buffer = io.BytesIO(image_data)
        img = ImageReader(img_buffer)
        img_w, img_h = img.getSize()
        aspect = img_h / float(img_w)
        
        draw_width = 500
        draw_height = draw_width * aspect
        
        # Max height check
        if draw_height > 600:
             draw_height = 600
             draw_width = draw_height / aspect

        x_pos = (width - draw_width) / 2
        y_pos = height - 120 - draw_height 
        
        can.drawImage(img, x_pos, y_pos, width=draw_width, height=draw_height)
    except Exception as e:
        can.drawString(50, height - 150, f"Error rendering image: {e}")

    can.save()
    packet.seek(0)
    
    # 4. Encrypt with OTP
    new_pdf = PdfReader(packet)
    writer = PdfWriter()
    
    for page in new_pdf.pages:
        writer.add_page(page)
        
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

if __name__ == '__main__':
    print("-------------------------------------------------------")
    print("GOVERNMENT SYSTEM ONLINE")
    print("Login Page: http://127.0.0.1:5000/login")
    print("-------------------------------------------------------")
    app.run(debug = True, port = 5000)