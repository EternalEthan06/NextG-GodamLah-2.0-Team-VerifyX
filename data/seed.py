
import sqlite3
import random
from faker import Faker
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash
from cryptography.fernet import Fernet # type: ignore

# Setup Faker and Encryption
fake = Faker() # Default Locale
cipher_suite = Fernet(Fernet.generate_key())

# Mock Encrypter Class (Simplified)
class MockEncrypter:
    def encrypt_cell(self, data):
        return cipher_suite.encrypt(data.encode()).decode()

MOCK_ENCRYPTER = MockEncrypter()

def create_database():
    conn = sqlite3.connect('data/verifyx.db')
    cursor = conn.cursor()

    # Drop old tables if exist
    cursor.execute('DROP TABLE IF EXISTS citizens')
    cursor.execute('DROP TABLE IF EXISTS access_logs') # New table for audit trail

    # Create Citizens Table
    cursor.execute('''
        CREATE TABLE citizens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT,
            mykad_number TEXT UNIQUE,
            address TEXT,
            income_bracket TEXT,
            oku_status TEXT,
            digital_signature TEXT,
            password_hash TEXT,
            birth_cert_enc TEXT,
            water_bill_enc TEXT,
            oku_status_enc TEXT,
            voice_audio_blob BLOB,
            face_image_blob BLOB,
            face_encoding_blob BLOB,
            failed_attempts INTEGER DEFAULT 0,
            lockout_until DATETIME DEFAULT NULL
        )
    ''')

    # Create Access Logs Table (Audit Trail)
    cursor.execute('''
        CREATE TABLE access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mykad_number TEXT,
            action TEXT, 
            context TEXT, 
            organization_name TEXT, 
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT
        )
    ''')
        
    # ---------------------------------------------------------
    # SEED DATA GENERATION
    # ---------------------------------------------------------
    
    citizens_data_for_db = []
    
    # 1. Known Demo User (You can login with this)
    known_mykad = "981225-14-5521"
    known_password = "password123"
    
    citizens_data_for_db.append((
        "Lee Jasmin", 
        known_mykad, 
        fake.address().replace('\n', ', '), 
        "T20", 
        "Active", 
        fake.sha256(),
        generate_password_hash(known_password),
        MOCK_ENCRYPTER.encrypt_cell(f"BC-{known_mykad} | Demo User"),
        MOCK_ENCRYPTER.encrypt_cell(f"ACCT-B5521-RM150 | Paid"),
        MOCK_ENCRYPTER.encrypt_cell("Active"),
        None,
        None,
        None
    ))

    # --- CUSTOM USERS REQUESTED ---
    # 2. Ethan Tiang Yong Xuan
    ethan_mykad = "020512-10-1234"
    citizens_data_for_db.append((
        "Ethan Tiang Yong Xuan", 
        ethan_mykad, 
        "123, Jalan Teknologi, Cyberjaya", 
        "T20", 
        "Active", 
        fake.sha256(),
        generate_password_hash("password123"), 
        MOCK_ENCRYPTER.encrypt_cell(f"BC-{ethan_mykad} | Ethan"),
        MOCK_ENCRYPTER.encrypt_cell(f"water_bill_ethan"),
        MOCK_ENCRYPTER.encrypt_cell("Active"),
        None,
        None,
        None
    ))

    # 3. Chloe Lai Pui Yan
    chloe_mykad = "021115-08-5678"
    citizens_data_for_db.append((
        "Chloe Lai Pui Yan", 
        chloe_mykad, 
        "456, Lorong Bunga, Penang", 
        "M40", 
        "Inactive", 
        fake.sha256(),
        generate_password_hash("password123"), 
        MOCK_ENCRYPTER.encrypt_cell(f"BC-{chloe_mykad} | Chloe"),
        MOCK_ENCRYPTER.encrypt_cell(f"water_bill_chloe"),
        MOCK_ENCRYPTER.encrypt_cell("Inactive"),
        None,
        None,
        None
    ))

    income_options = ["B40", "M40", "T20"]
    oku_options = ["Active", "Inactive"]

    # Generate 97 Random Citizens (Total = 1 Demo + 2 Custom + 97 = 100)
    for i in range(97):
        name = fake.name()
        mykad = fake.unique.numerify(text="######-##-####")
        
        # Ensure unique mykad doesn't clash with custom ones (simple check)
        while mykad == known_mykad or mykad == ethan_mykad or mykad == chloe_mykad:
             mykad = fake.unique.numerify(text="######-##-####")

        raw_password = f"user{i+1:03d}" 
        password_storage_hash = generate_password_hash(raw_password)
        address = fake.address().replace('\n', ', ')
        income = random.choice(income_options)
        oku = random.choice(oku_options)
        signature = fake.sha256()

        birth_cert_data = MOCK_ENCRYPTER.encrypt_cell(f"BC-{mykad} | Issued: JPN")
        water_bill_data = MOCK_ENCRYPTER.encrypt_cell(f"ACCT-B{mykad[-4:]}-RM150 | Status: Paid")
        oku_data = MOCK_ENCRYPTER.encrypt_cell(oku)

        citizens_data_for_db.append((name, mykad, address, income, oku, signature, password_storage_hash, birth_cert_data, water_bill_data, oku_data, None, None, None))


    # Insert citizen data
    cursor.executemany('''
        INSERT INTO citizens (full_name, mykad_number, address, income_bracket, oku_status, digital_signature, 
                              password_hash, birth_cert_enc, water_bill_enc, oku_status_enc, voice_audio_blob, face_image_blob, face_encoding_blob)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', citizens_data_for_db)
    
    # 5. Insert Mock Access Log Data (Populates the new access_logs table)
    now = datetime.now()
    logs_data_for_db = [
        # Demo User Logs
        (known_mykad, 'SHARE', 'Identity Card & Income Slip (P2V)', 'Speedhome Rental', (now - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'), 'ACTIVE'),
        (known_mykad, 'SHARE', 'Income Bracket (M40 Status)', 'Public Bank Berhad', (now - timedelta(minutes=45)).strftime('%Y-%m-%d %H:%M:%S'), 'ACTIVE'),
        (known_mykad, 'VERIFY', 'Facial ID (eKYC)', 'Touch \'n Go eWallet', (now - timedelta(hours=5)).strftime('%Y-%m-%d %H:%M:%S'), 'REVOKED'),
        (known_mykad, 'SHARE', 'Tax Clearance (2024)', 'LHDN Malaysia', (now - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'), 'EXPIRED'),
        (known_mykad, 'ACCESS', 'Failed Biometric Challenge', 'Unknown Device (192.168.1.55)', (now - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S'), 'BLOCKED'),
        (known_mykad, 'LOGIN', 'Portal Access', 'System', (now - timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S'), 'SUCCESS'),

        # Ethan Logs (T20 Profile)
        (ethan_mykad, 'SHARE', 'Income Tax Statement (2024)', 'LHDN E-Filing', (now - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S'), 'ACTIVE'),
        (ethan_mykad, 'VERIFY', 'Voice Biometric (High Value)', 'Maybank2u Premier', (now - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'), 'SUCCESS'),
        (ethan_mykad, 'SHARE', 'Property Ownership Proof', 'MBSJ Council', (now - timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S'), 'ACTIVE'),
        (ethan_mykad, 'LOGIN', 'Biometric Login', 'System', (now - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'), 'SUCCESS'),

        # Chloe Logs (M40 Profile)
        (chloe_mykad, 'SHARE', 'University Transcript', 'JobStreet Profile', (now - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S'), 'ACTIVE'),
        (chloe_mykad, 'VERIFY', 'Face ID Verification', 'GrabPay E-Wallet', (now - timedelta(hours=4)).strftime('%Y-%m-%d %H:%M:%S'), 'SUCCESS'),
        (chloe_mykad, 'SHARE', 'Bantuan Prihatin Status', 'MySejahtera', (now - timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S'), 'REVOKED'),
        (chloe_mykad, 'LOGIN', 'Portal Access', 'System', (now - timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S'), 'SUCCESS'),
    ]

    cursor.executemany('''
        INSERT INTO access_logs (mykad_number, action, context, organization_name, timestamp, status)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', logs_data_for_db)


    conn.commit()
    conn.close()
    
    print("\n" + "="*50)
    print("✅ DATABASE READY: 'verifyx.db' populated securely.")
    print("✅ access_logs table created and populated with mock data.")
    print("==================================================")
    print(f"🚨 JASMIN CREDENTIALS: MYKAD: {known_mykad}, PASSWORD: {known_password}")
    print(f"🚨 ETHAN CREDENTIALS: MYKAD: {ethan_mykad}, PASSWORD: password123")
    print(f"🚨 CHLOE CREDENTIALS: MYKAD: {chloe_mykad}, PASSWORD: password123")
    
if __name__ == "__main__":
    create_database()