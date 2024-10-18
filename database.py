import sqlite3
from datetime import datetime

def connect_to_database():
    return sqlite3.connect('predictions.db')

# Create a table to store the prdictions 
def create_table(conn):
    c = conn.cursor()
    c.execute('''DROP TABLE IF EXISTS predictions''')
    c.execute('''CREATE TABLE predictions
                 (student_name TEXT, predicted_result REAL, gpa REAL,timestamp TEXT)''')
    conn.commit()
# Insert the predictions to the table 
def insert_prediction(conn, student_name, predicted_result, gpa):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c = conn.cursor()
    c.execute("INSERT INTO predictions (student_name, predicted_result, gpa, timestamp) VALUES ( ?, ?, ?, ?)",
              (student_name, predicted_result, gpa, timestamp))
    conn.commit()


























