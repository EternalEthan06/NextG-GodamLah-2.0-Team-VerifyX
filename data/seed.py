# File: data/seed.py (FINAL SECURE VERSION WITH LOGS AND HASHING)

import sqlite3
from faker import Faker
import random
import os
from cryptography.fernet import Fernet
from werkzeug.security import generate_password_hash 
from datetime import datetime, timedelta

# --- ENCRYPTION SETUP (Remains the same) ---
class SeedEncryptionEngine:
    # Using a fixed key for seeding consistency, but Fernet.generate_key() is better for production
    def __init__(self):
        # Using a fixed key for consistent mock encrypted data
        self.__key = os.getenv("KMS_SEED_KEY", b'g_98yC-Y9Q3uRzD7aDk4bK9jH8sL7w2T-rE2qF4xZ1w=')
        self.__cipher = Fernet(self.__key)
    def encrypt_cell(self, data: str) -> str:
        if not data: return ""
        try:
            return self.__cipher.encrypt(data.encode()).decode()
        except Exception:
            return ""

MOCK_ENCRYPTER = SeedEncryptionEngine()
fake = Faker()

def create_database():
    conn = sqlite3.connect('data/verifyx.db')
    cursor = conn.cursor()

    # 1. Drop existing tables for clean seed
    cursor.execute('''DROP TABLE IF EXISTS citizens;''')
    cursor.execute('''DROP TABLE IF EXISTS access_logs;''') # <-- CRITICAL: Dropping the old/missing logs table

    # 2. Create the Citizens Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS citizens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT,
            mykad_number TEXT,
            address TEXT,
            income_bracket TEXT,
            oku_status TEXT,
            digital_signature TEXT,
            password_hash TEXT,
            birth_cert_enc TEXT,
            water_bill_enc TEXT,
            oku_status_enc TEXT
        )
    ''')

    # 3. Create the Access Logs Table (NEW SCHEMA to fix OperationalError)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mykad_number TEXT NOT NULL,
            action TEXT NOT NULL,
            context TEXT,
            organization_name TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT
        )
    ''')

    # 4. Generate 100 Mock Citizens
    print("Generating 100 citizens... Please wait.")
    
    income_options = ["B40", "M40", "T20"]
    oku_options = ["Active", "Not Applicable"]
    
    citizens_data_for_db = []
    
    known_mykad = "981225-14-5521" 
    known_password = "password123"
    
    # Generate the first user record manually
    citizens_data_for_db.append((
        "Lee Jasmin (DEMO)", 
        known_mykad, 
        fake.address().replace('\n', ', '), 
        "T20", 
        "Active", 
        fake.sha256(),
        generate_password_hash(known_password),
        MOCK_ENCRYPTER.encrypt_cell(f"BC-{known_mykad} | Demo User"),
        MOCK_ENCRYPTER.encrypt_cell(f"ACCT-B5521-RM150 | Paid"),
        MOCK_ENCRYPTER.encrypt_cell("Active")
    ))

    for i in range(99):
        name = fake.name()
        mykad = f"{random.randint(50, 99)}{random.randint(10, 12)}{random.randint(10, 30)}-{random.randint(10, 14)}-{random.randint(1000, 9999)}"
        raw_password = f"user{i+1:03d}" 
        password_storage_hash = generate_password_hash(raw_password)
        address = fake.address().replace('\n', ', ')
        income = random.choice(income_options)
        oku = random.choice(oku_options)
        signature = fake.sha256()

        birth_cert_data = MOCK_ENCRYPTER.encrypt_cell(f"BC-{mykad} | Issued: JPN")
        water_bill_data = MOCK_ENCRYPTER.encrypt_cell(f"ACCT-B{mykad[-4:]}-RM150 | Status: Paid")
        oku_data = MOCK_ENCRYPTER.encrypt_cell(oku)

        citizens_data_for_db.append((name, mykad, address, income, oku, signature, password_storage_hash, birth_cert_data, water_bill_data, oku_data))


    # Insert citizen data
    cursor.executemany('''
        INSERT INTO citizens (full_name, mykad_number, address, income_bracket, oku_status, digital_signature, 
                              password_hash, birth_cert_enc, water_bill_enc, oku_status_enc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', citizens_data_for_db)
    
    # 5. Insert Mock Access Log Data (Populates the new access_logs table)
    now = datetime.now()
    logs_data_for_db = [
        (known_mykad, 'SHARE', 'Identity Card & Income Slip (P2V)', 'Speedhome Rental', (now - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'), 'ACTIVE'),
        (known_mykad, 'SHARE', 'Income Bracket (M40 Status)', 'Public Bank Berhad', (now - timedelta(minutes=45)).strftime('%Y-%m-%d %H:%M:%S'), 'ACTIVE'),
        (known_mykad, 'VERIFY', 'Facial ID (eKYC)', 'Touch \'n Go eWallet', (now - timedelta(hours=5)).strftime('%Y-%m-%d %H:%M:%S'), 'REVOKED'),
        (known_mykad, 'SHARE', 'Tax Clearance (2024)', 'LHDN Malaysia', (now - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'), 'EXPIRED'),
        (known_mykad, 'ACCESS', 'Failed Biometric Challenge', 'Unknown Device (192.168.1.55)', (now - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S'), 'BLOCKED'),
        (known_mykad, 'LOGIN', 'Portal Access', 'System', (now - timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S'), 'SUCCESS'),
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
    print(f"🚨 DEMO USER CREDENTIALS (FOR TESTING ONLY): MYKAD: {known_mykad}, PASSWORD: {known_password}")
    
if __name__ == "__main__":
    create_database()