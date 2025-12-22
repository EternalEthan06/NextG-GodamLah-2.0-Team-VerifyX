import sqlite3 
import datetime 
import time 
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
import pickle

# Initialize Flask
app = Flask(__name__, template_folder='Front-End/templates')

# --- BIOMETRIC MODEL INITIALIZATION ---
# Load the model once at startup (this might take a few seconds)
print("Loading Voice Recognition Model...")
encoder = VoiceEncoder()
print("Voice Model Loaded.")

# Configuration
app.secret_key = 'your_super_secret_and_unique_key_12345' 
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(minutes=5) # Auto-logout after 5 mins
UPLOAD_FOLDER = 'uploads'
DATABASE = os.path.join(os.path.dirname(__file__), 'data', 'verifyx.db')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok = True)
os.makedirs(os.path.dirname(DATABASE), exist_ok = True)

# --- AUTO-INITIALIZE DATABASE ---
if not os.path.exists(DATABASE):
    print("WARNING: Database not found. Initializing new database from seed...")
    try:
        from data import seed
        seed.create_database()
        print("SUCCESS: Database initialized.")
    except Exception as e:
        print(f"ERROR: Could not initialize database: {e}")


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

# --- MEDIAPIPE INITIALIZATION ---
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

import io

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

# --- END MEDIAPIPE ---

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
    mykad = data.get('ic-number')
    password = data.get('password')

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
            if user_record['voice_audio_blob']:
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

@app.route('/verify-voice', methods=['POST'])
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
    recent_logs = fetch_access_logs(mykad)[:5]
    
    return render_template('dashboard.html', user=user_data, logs=recent_logs)

@app.route('/files')
def my_files_entry():
    if not session.get('vault_access_granted'):
        return redirect(url_for('dashboard', alert='verification_failed'))
    
    user_data, mykad = get_user_context()
    if not user_data: 
        return redirect(url_for('login'))
    
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
    else:
        oku_status = 'Not Applicable'

    files_list.append({
        'name': 'OKU Status Card',
        'mykad_link': user_data['oku_status_enc'],
        'status': oku_status,
        'icon': 'fa-wheelchair'
    })    

    return render_template('allFiles.html', user = user_data, files = files_list)

@app.route('/share')
def share():
    """
    Displays the user's active shared access records, allowing the user to
    view or manage documents they have shared with others.
    """
    if not session.get('vault_access_granted'): 
        return redirect(url_for('login'))
    user_data, mykad = get_user_context()
    active_shares = fetch_access_logs(mykad, log_status='ACTIVE')
    return render_template('share.html', user = user_data, active_shares = active_shares)

@app.route('/logs')
def access_log(): 
    """
    Displays all access logs for the currently logged-in user.
    """
    if not session.get('vault_access_granted'): 
        return redirect(url_for('login'))
    user_data, mykad = get_user_context()
    all_logs = fetch_access_logs(mykad)
    return render_template('accessLog.html', user = user_data, logs = all_logs)

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

@app.route('/logout')
def logout():
    """
    Logs out the current user by clearing all session data.
    """
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    print("-------------------------------------------------------")
    print("GOVERNMENT SYSTEM ONLINE")
    print("Login Page: http://127.0.0.1:5000/login")
    print("-------------------------------------------------------")
    app.run(debug = True, port = 5000)