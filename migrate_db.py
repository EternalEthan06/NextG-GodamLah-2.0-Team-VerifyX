import sqlite3

def migrate():
    db_path = 'data/verifyx.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"Connected to {db_path}")

    # Check existing columns
    cursor.execute('PRAGMA table_info(citizens)')
    existing_columns = [row[1] for row in cursor.fetchall()]
    print(f"Existing columns: {existing_columns}")

    # Columns to add
    columns_to_add = {
        'failed_attempts': 'INTEGER DEFAULT 0',
        'lockout_until': 'TEXT',
        'face_image_blob': 'BLOB',
        'face_encoding_blob': 'BLOB'
    }

    for col_name, col_type in columns_to_add.items():
        if col_name not in existing_columns:
            print(f"Adding column: {col_name} ({col_type})")
            try:
                cursor.execute(f'ALTER TABLE citizens ADD COLUMN {col_name} {col_type}')
                print("  - Success")
            except sqlite3.OperationalError as e:
                print(f"  - Failed: {e}")
        else:
            print(f"Column {col_name} already exists.")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
