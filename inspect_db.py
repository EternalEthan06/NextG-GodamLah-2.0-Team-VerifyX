import sqlite3

try:
    conn = sqlite3.connect('data/verifyx.db')
    cursor = conn.cursor()
    cursor.execute('PRAGMA table_info(citizens)')
    columns = [row[1] for row in cursor.fetchall()]
    print("Columns in citizens table:")
    print(columns)
    conn.close()
except Exception as e:
    print(e)
