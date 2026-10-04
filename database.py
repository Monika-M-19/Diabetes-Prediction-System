import sqlite3
from pathlib import Path
from datetime import datetime


# ============================================================
# DATABASE LOCATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_FILE = BASE_DIR / "predictions.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    connection = sqlite3.connect(DATABASE_FILE)

    return connection


# ============================================================
# CREATE DATABASE TABLE
# ============================================================

def init_db():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            pregnancies REAL NOT NULL,

            glucose REAL NOT NULL,

            blood_pressure REAL NOT NULL,

            skin_thickness REAL NOT NULL,

            insulin REAL NOT NULL,

            bmi REAL NOT NULL,

            diabetes_pedigree REAL NOT NULL,

            age REAL NOT NULL,

            prediction TEXT NOT NULL,

            probability REAL NOT NULL,

            created_at TEXT NOT NULL

        )
    """)

    connection.commit()

    connection.close()


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age,
    prediction,
    probability
):

    connection = get_connection()

    cursor = connection.cursor()

    created_at = datetime.now().strftime(
        "%d-%m-%Y %H:%M"
    )

    cursor.execute("""
        INSERT INTO predictions (

            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age,
            prediction,
            probability,
            created_at

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (

        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age,
        prediction,
        probability,
        created_at

    ))

    connection.commit()

    connection.close()


# ============================================================
# GET PREDICTION HISTORY
# ============================================================

def get_predictions():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        SELECT

            glucose,
            bmi,
            age,
            prediction,
            probability,
            created_at

        FROM predictions

        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    connection.close()

    history = []

    for row in rows:

        history.append({

            "Glucose": row[0],
            "BMI": row[1],
            "Age": row[2],
            "Prediction": row[3],
            "Probability": row[4],
            "Date": row[5]

        })

    return history


# ============================================================
# CLEAR ALL HISTORY
# ============================================================

def clear_predictions():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM predictions
    """)

    connection.commit()

    connection.close()