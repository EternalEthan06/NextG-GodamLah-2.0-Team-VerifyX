import os
import sqlite3 
import datetime 
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash 

# Initialize Flask
app = Flask(__name__, template_folder='Front-End/templates')

# Configuration
app.secret_key = 'your_super_secret_and_unique_key_12345' 
UPLOAD_FOLDER = 'uploads'
DATABASE = os.path.join(os.path.dirname(__file__), 'data', 'verifyx.db')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok = True)
os.makedirs(os.path.dirname(DATABASE), exist_ok = True)

USER_ENROLLED = False

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

# LOGIN & BIOMETRIC LOGIC
@app.route('/login')
def login():
    """
    Handles the login page route.    
    """
    if session.get('vault_access_granted'):
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/handle_login', methods = ['POST'])
def handle_login():
    """
    Handles login requests by verifying MyKad and password, 
    then directs users to biometric verification or enrollment if needed.
    """
    data = request.get_json() 
    mykad = data.get('ic-number')
    password = data.get('password')

    user_record = fetch_user_record_by_mykad(mykad)
    
    if user_record:
        stored_hash = user_record['password_hash']
        if stored_hash and check_password_hash(stored_hash, password):
            
            # Store Identity Temporarily
            session['temp_mykad'] = mykad
            session['user_name'] = user_record['full_name'].split()[0]
            session['vault_access_granted'] = False 
            
            # Redirect to The Biometric Gateway
            if not USER_ENROLLED:
                session['enrollment_stage'] = 'face'
                log_action(mykad, "LOGIN_ATTEMPT", "New User Registration", "System", "PENDING")
            else:
                log_action(mykad, "LOGIN_ATTEMPT", "Biometric Verification", "System", "PENDING")
            
            # Success: Go to the Gateway
            return jsonify({'status': 'success', 'redirect': url_for('verify_identity')})
    
    log_action(mykad or 'UNKNOWN', "LOGIN", "Portal Access", "System", "FAILED")
    return jsonify({'status': 'failure', 'message': 'Invalid MyKad or Password.'}), 401

@app.route('/verify-identity')
def verify_identity():
    """
    Directs users to the appropriate biometric verification or enrollment step
    based on their session and enrollment status.
    """
    if 'temp_mykad' not in session: 
        return redirect(url_for('login'))
    
    if not USER_ENROLLED:
        session['enrollment_stage'] = 'face'
        return redirect(url_for('biometric_step'))
    else:
        return redirect(url_for('select_biometric'))

@app.route('/select-biometric')
def select_biometric():
    """
    Displays the biometric selection page for users who have started the login process.
    """
    if 'temp_mykad' not in session: 
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
    return redirect(url_for('step_complete')) 

@app.route('/biometric-step')
def biometric_step():
    """
    Directs users to the correct biometric enrollment or verification step 
    based on their current stage stored in the session.
    """
    if 'temp_mykad' not in session: 
        return redirect(url_for('login'))
    
    stage = session.get('enrollment_stage')
    if stage == 'face': 
        return render_template('faceScan.html') 
    elif stage == 'voice': 
        return render_template('voiceTest.html') 
    elif stage == 'gesture': 
        return render_template('gestureTest.html') 
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
    if 'temp_mykad' not in session: return redirect(url_for('login'))
    
    mykad = session['temp_mykad']
    
    # LOGIC A: REGISTRATION
    if not USER_ENROLLED:
        current = session.get('enrollment_stage')
        if current == 'face':
            session['enrollment_stage'] = 'voice'
            return redirect(url_for('biometric_step'))
        elif current == 'voice':
            session['enrollment_stage'] = 'gesture'
            return redirect(url_for('biometric_step'))
        elif current == 'gesture':
            USER_ENROLLED = True
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

    return redirect(url_for('login'))

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
        print(f"WARNING: No user found for MyKad: {mykad}")
        session.clear()  # Clear invalid session
        return redirect(url_for('login'))
    
    return render_template('dashboard.html', user=user_data)

@app.route('/files')
def my_files_entry():
    """
    Displays the user's personal files dashboard, including documents like
    birth certificate, water bill and OKU status card.
    """
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