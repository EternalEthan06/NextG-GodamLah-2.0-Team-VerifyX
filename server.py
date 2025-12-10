import os
import sqlite3 
import datetime 
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash # Keep generate_password_hash for the mock/temp fix

# Initialize Flask
app = Flask(__name__, template_folder='Front-End/templates')

# --- CONFIGURATION FIX ---
app.secret_key = 'your_super_secret_and_unique_key_12345' 
# -------------------------

UPLOAD_FOLDER = 'uploads'
DATABASE = 'data/verifyx.db' # Database path
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Create the upload folder and database directory if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(DATABASE), exist_ok=True)

USER_ENROLLED = False

# Helper function to check file types
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==========================================
#         DATABASE ACCESS LAYER (NEW)
# ==========================================

def get_db_connection():
    """Establishes an SQLite connection and configures row factory."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name (e.g., row['full_name'])
    return conn

def fetch_user_record_by_mykad(mykad):
    """Fetches a citizen's record using their MyKad number."""
    try:
        conn = get_db_connection()
        user_record = conn.execute('SELECT * FROM citizens WHERE mykad_number = ?', (mykad,)).fetchone()
        conn.close()
        return user_record
    except sqlite3.OperationalError as e:
        print(f"DATABASE ERROR: {e}")
        # Return None if the table is missing or DB is locked
        return None 

def log_action(mykad, action, context, organization_name, status="SUCCESS"):
    """Inserts a new audit log entry."""
    try:
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO access_logs (mykad_number, action, context, organization_name, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (mykad, action, context, organization_name, status))
        conn.commit()
        conn.close()
    except sqlite3.OperationalError as e:
        print(f"LOGGING ERROR: Could not write to access_logs. Did you run seed.py? Error: {e}")

def fetch_access_logs(mykad, log_status=None):
    """Fetches audit logs for a specific user, optionally filtered by status."""
    try:
        conn = get_db_connection()
        query = 'SELECT * FROM access_logs WHERE mykad_number = ? ORDER BY timestamp DESC'
        params = (mykad,)
        
        if log_status:
            query = 'SELECT * FROM access_logs WHERE mykad_number = ? AND status = ? ORDER BY timestamp DESC'
            params = (mykad, log_status)
            
        logs = conn.execute(query, params).fetchall()
        conn.close()
        
        # Convert Row objects to dictionaries for easier Jinja templating
        return [dict(log) for log in logs]
    except sqlite3.OperationalError as e:
        print(f"FETCH LOGS ERROR: Could not read from access_logs. Did you run seed.py? Error: {e}")
        return []

def get_user_context():
    """Helper to get user data and handle unauthenticated access."""
    mykad = session.get('user_mykad')
    if not mykad:
        return None, None
    user_record = fetch_user_record_by_mykad(mykad)
    
    if user_record:
        # Convert Row to dict and add first name for sidebar display
        user_dict = dict(user_record) 
        user_dict['first_name'] = user_dict['full_name'].split()[0]
        return user_dict, mykad
    return None, None

# ==========================================
#              PAGE ROUTES
# ==========================================

# 1. LOGIN PAGE (Entry Point)
@app.route('/login')
def login():
    return render_template('login.html')

# 1B. LOGIN HANDLER (POST request from the form)
@app.route('/handle_login', methods=['POST'])
def handle_login():
    data = request.get_json() 
    mykad = data.get('ic-number')
    password = data.get('password')

    # 1. Fetch User Record (Using the actual DB function)
    user_record = fetch_user_record_by_mykad(mykad)
    
    if user_record:
        # --- FIX 1: Access column via square brackets [] instead of .get() ---
        stored_hash = user_record['password_hash']
        # --------------------------------------------------------------------
        
        # 2. Authentication Check: Verify password against the secure hash
        if stored_hash and check_password_hash(stored_hash, password):
            
            # Authentication Success: Store MyKad and name in session
            session['user_mykad'] = mykad
            # Access full_name via square brackets []
            session['user_name'] = user_record['full_name'].split()[0] 
            
            # 3. Log Audit Trail 
            log_action(mykad, "LOGIN", "Portal Access", "System", "SUCCESS")
            
            return jsonify({'status': 'success', 'redirect': url_for('dashboard')})
    
    # If user not found OR hash check failed 
    log_action(mykad or 'UNKNOWN', "LOGIN", "Portal Access", "System", "FAILED")
    return jsonify({'status': 'failure', 'message': 'Invalid MyKad or Password.'}), 401

# 2. DASHBOARD (Main Home)
@app.route('/')
@app.route('/dashboard')
def dashboard():
    user_data, mykad = get_user_context()
    if not user_data:
        return redirect(url_for('login'))
        
    return render_template('dashboard.html', user=user_data) 

# --- LOGOUT ROUTE ---
@app.route('/logout')
def logout():
    mykad = session.get('user_mykad', 'UNKNOWN')
    log_action(mykad, "LOGOUT", "Portal Exit", "System", "SUCCESS")
    
    session.pop('user_mykad', None)
    session.pop('user_name', None)
    return redirect(url_for('login'))

# --- LOGS ROUTE (accessLog.html integration) ---
@app.route('/logs')
def access_log():
    user_data, mykad = get_user_context()
    if not user_data:
        return redirect(url_for('login'))
        
    all_logs = fetch_access_logs(mykad)
    
    return render_template('accessLog.html', user=user_data, logs=all_logs) 

# --- SHARE ROUTE (share.html integration) ---
@app.route('/share')
def share():
    user_data, mykad = get_user_context()
    if not user_data:
        return redirect(url_for('login'))
        
    active_shares = fetch_access_logs(mykad, log_status='ACTIVE')
    
    return render_template('share.html', user=user_data, active_shares=active_shares)

# --- NEW: Route to handle revocation of access (Used by accessLog.html and share.html) ---
@app.route('/revoke_access/<int:log_id>', methods=['POST'])
def revoke_access(log_id):
    if 'user_mykad' not in session:
        return jsonify({'status': 'failure', 'message': 'Unauthorized'}), 401

    mykad = session.get('user_mykad')
    conn = get_db_connection()
    
    # 1. Update the log status to 'REVOKED'
    try:
        cursor = conn.execute('''
            UPDATE access_logs SET status = 'REVOKED' 
            WHERE id = ? AND mykad_number = ? AND status = 'ACTIVE'
        ''', (log_id, mykad))
    except sqlite3.OperationalError as e:
        conn.close()
        return jsonify({'status': 'failure', 'message': f'Database Error: {e}'}), 500
        
    if cursor.rowcount == 0:
         conn.close()
         return jsonify({'status': 'failure', 'message': 'Access not found or already terminated.'}), 404
         
    conn.commit()
    conn.close()
    
    # 2. Log the revocation action
    log_action(mykad, "REVOKE", f"Revoked Log ID: {log_id}", "User Action", "SUCCESS")
    
    return jsonify({'status': 'success', 'message': 'Access revoked successfully!'})


# --- SETTINGS ROUTE (FIXED ENDPOINT NAME) ---
@app.route('/settings')
def settings():
    user_data, mykad = get_user_context()
    if not user_data:
        return redirect(url_for('login'))
        
    return render_template('settings.html', user=user_data) 

@app.route('/files')
def my_files_entry():
    global USER_ENROLLED
    
    if 'user_mykad' not in session:
        return redirect(url_for('login'))
    
    session['vault_access_granted'] = False
    
    if not USER_ENROLLED:
        session['enrollment_stage'] = 'face' 
        return redirect(url_for('biometric_step'))
    else:
        return render_template('select_biometric.html')

# --- ROUTER: Loads the correct HTML page (Untouched) ---
@app.route('/biometric-step')
def biometric_step():
    stage = session.get('enrollment_stage')
    
    if stage == 'face':
        return render_template('faceScan.html') 
    elif stage == 'voice':
        return render_template('voiceTest.html') 
    elif stage == 'gesture':
        return render_template('gestureTest.html') 
    else:
        return redirect(url_for('dashboard', alert='system_error'))

# --- STEP COMPLETED HANDLER (Untouched) ---
@app.route('/step-complete')
def step_complete():
    global USER_ENROLLED
    
    if not USER_ENROLLED:
        current_stage = session.get('enrollment_stage')
        
        if current_stage == 'face':
            session['enrollment_stage'] = 'voice'
            return redirect(url_for('biometric_step'))
            
        elif current_stage == 'voice':
            session['enrollment_stage'] = 'gesture'
            return redirect(url_for('biometric_step'))
            
        elif current_stage == 'gesture':
            USER_ENROLLED = True 
            session['vault_access_granted'] = True
            session.pop('enrollment_stage', None)
            return redirect(url_for('final_vault'))

    else:
        queue = session.get('verification_queue', [])
        
        if len(queue) > 0:
            next_method = queue.pop(0)
            session['verification_queue'] = queue 
            session['enrollment_stage'] = next_method 
            return redirect(url_for('biometric_step'))
        else:
            session['vault_access_granted'] = True
            return redirect(url_for('final_vault'))

    return redirect(url_for('dashboard'))

# --- RETURNING USER SELECTION HANDLER (Untouched) ---
@app.route('/verify-selection', methods=['POST'])
def verify_selection():
    methods = request.form.getlist('methods') 
    
    if len(methods) != 2:
        return "Error: Please select exactly 2 methods."
    
    session['verification_queue'] = methods
    return redirect(url_for('step_complete')) 

# --- FINAL VAULT (Protected: allFiles.html integration) ---
@app.route('/files-vault')
def final_vault():
    if not session.get('vault_access_granted'):
        return redirect(url_for('dashboard', alert='verification_failed'))
    
    user_data, mykad = get_user_context()
    if not user_data:
        return redirect(url_for('login'))
        
    # Map the encrypted columns to user-friendly file data for allFiles.html
    files_list = []
    
    # Access encrypted data fields via square brackets []
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
    files_list.append({
        'name': 'OKU Status Card', 
        'mykad_link': user_data['oku_status_enc'], 
        'status': 'Active' if user_data['oku_status'] == 'Active' else 'Not Applicable',
        'icon': 'fa-wheelchair'
    })
    
    return render_template('allFiles.html', user=user_data, files=files_list) 

# ==========================================
#              START SERVER
# ==========================================
if __name__ == '__main__':
    print("-------------------------------------------------------")
    print("🏛️  GOVERNMENT SYSTEM ONLINE")
    print("-------------------------------------------------------")
    print(" -> Login Page:   http://127.0.0.1:5000/login")
    print(" -> Dashboard:    http://127.0.0.1:5000/")
    print("-------------------------------------------------------")
    
    app.run(debug=True, port=5000)