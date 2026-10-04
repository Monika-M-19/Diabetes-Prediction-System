import gradio as gr
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# DATABASE
# ============================================================

from database import (
    init_db,
    save_prediction,
    get_predictions
)

# ============================================================
# INITIALIZE DATABASE
# ============================================================

init_db()

# ============================================================
# LOAD MODEL FILES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "diabetes_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
imputer = joblib.load(BASE_DIR / "imputer.pkl")


# ============================================================
# HISTORY HTML
# ============================================================

def get_history():

    prediction_history = get_predictions()

    if not prediction_history:

        return """
        <div class="empty-history">

            <div class="empty-icon">
                ◷
            </div>

            <div class="empty-title">
                No predictions yet
            </div>

            <div class="empty-text">
                Your prediction history will appear here.
            </div>

        </div>
        """

    rows = ""

    for item in prediction_history:

        if item["Prediction"] == "Diabetes":

            badge = (
                '<span class="badge diabetes-badge">'
                'Diabetes'
                '</span>'
            )

        else:

            badge = (
                '<span class="badge healthy-badge">'
                'No Diabetes'
                '</span>'
            )

        rows += f"""
        <tr>

            <td>{item["Glucose"]}</td>

            <td>{item["BMI"]}</td>

            <td>{item["Age"]}</td>

            <td>{badge}</td>

            <td>{item["Probability"]:.2f}%</td>

            <td>{item["Date"]}</td>

        </tr>
        """

    return f"""
    <div class="history-table-container">

        <table class="history-table">

            <thead>

                <tr>

                    <th>Glucose</th>
                    <th>BMI</th>
                    <th>Age</th>
                    <th>Prediction</th>
                    <th>Probability</th>
                    <th>Date & Time</th>

                </tr>

            </thead>

            <tbody>

                {rows}

            </tbody>

        </table>

    </div>
    """


# ============================================================
# DEFAULT RESULT
# ============================================================

def default_result():

    result_html = """
    <div class="result-content neutral-result">

        <div class="result-icon">
            ✦
        </div>

        <div class="result-status">
            READY
        </div>

        <div class="result-title">
            Awaiting Prediction
        </div>

        <div class="result-description">

            Enter the patient's information and click
            <b>Predict Diabetes</b> to see the result.

        </div>

    </div>
    """

    probability_html = """
    <div class="probability-box">

        <div class="probability-top">

            <div class="probability-label">
                Diabetes Probability
            </div>

            <strong class="probability-value">
                0.00%
            </strong>

        </div>

        <div class="progress-background">

            <div
                class="progress-fill"
                style="width:0%;">
            </div>

        </div>

        <div class="probability-info">

            Probability will be calculated by the
            Logistic Regression model.

        </div>

    </div>
    """

    return result_html, probability_html


# ============================================================
# EMPTY PROBABILITY
# ============================================================

def empty_probability(message):

    return f"""
    <div class="probability-box">

        <div class="probability-top">

            <div class="probability-label">
                Diabetes Probability
            </div>

            <strong class="probability-value">
                0.00%
            </strong>

        </div>

        <div class="progress-background">

            <div
                class="progress-fill"
                style="width:0%;">
            </div>

        </div>

        <div class="probability-info">

            {message}

        </div>

    </div>
    """


# ============================================================
# WARNING RESULT
# ============================================================

def warning_result(title, description):

    return f"""
    <div class="result-content warning-result">

        <div class="result-icon">
            !
        </div>

        <div class="result-status">
            INPUT CHECK
        </div>

        <div class="result-title">
            {title}
        </div>

        <div class="result-description">
            {description}
        </div>

    </div>
    """


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
):

    values = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]

    # ========================================================
    # EMPTY FIELD CHECK
    # ========================================================

    if any(
        str(value).strip() == ""
        for value in values
    ):

        return (

            warning_result(
                "Complete all fields",
                "Please enter a value in all 8 patient information fields."
            ),

            empty_probability(
                "All patient information fields are required."
            ),

            get_history()

        )

    # ========================================================
    # CONVERT VALUES
    # ========================================================

    try:

        input_values = [
            float(pregnancies),
            float(glucose),
            float(blood_pressure),
            float(skin_thickness),
            float(insulin),
            float(bmi),
            float(diabetes_pedigree),
            float(age)
        ]

    except (ValueError, TypeError):

        return (

            warning_result(
                "Invalid input",
                "Please enter valid numeric values in all fields."
            ),

            empty_probability(
                "Only numeric values are accepted."
            ),

            get_history()

        )

    # ========================================================
    # NEGATIVE CHECK
    # ========================================================

    if any(
        value < 0
        for value in input_values
    ):

        return (

            warning_result(
                "Invalid values",
                "Negative values are not allowed."
            ),

            empty_probability(
                "Please enter non-negative values."
            ),

            get_history()

        )

    # ========================================================
    # AGE CHECK
    # ========================================================

    if input_values[7] <= 0:

        return (

            warning_result(
                "Invalid age",
                "Age must be greater than zero."
            ),

            empty_probability(
                "Please enter a valid age."
            ),

            get_history()

        )

    # ========================================================
    # CREATE INPUT ARRAY
    # ========================================================

    input_data = np.array(
        [input_values],
        dtype=float
    )

    # ========================================================
    # REPLACE INVALID ZERO VALUES
    # ========================================================

    invalid_zero_indices = [
        1,
        2,
        3,
        4,
        5
    ]

    for index in invalid_zero_indices:

        if input_data[0, index] == 0:

            input_data[0, index] = np.nan

    # ========================================================
    # IMPUTATION
    # ========================================================

    input_data = imputer.transform(
        input_data
    )

    # ========================================================
    # SCALING
    # ========================================================

    input_data = scaler.transform(
        input_data
    )

    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    prediction = model.predict(
        input_data
    )[0]

    probability = (
        model.predict_proba(input_data)[0][1]
        * 100
    )

    # ========================================================
    # RESULT
    # ========================================================

    if prediction == 1:

        prediction_text = "Diabetes"

        result_html = """
        <div class="result-content diabetes-result">

            <div class="result-icon">
                +
            </div>

            <div class="result-status">
                PREDICTION RESULT
            </div>

            <div class="result-title">
                Diabetes
            </div>

            <div class="result-description">

                The model predicts a higher likelihood of
                diabetes based on the entered information.

            </div>

        </div>
        """

    else:

        prediction_text = "No Diabetes"

        result_html = """
        <div class="result-content healthy-result">

            <div class="result-icon">
                ✓
            </div>

            <div class="result-status">
                PREDICTION RESULT
            </div>

            <div class="result-title">
                No Diabetes
            </div>

            <div class="result-description">

                The model predicts a lower likelihood of
                diabetes based on the entered information.

            </div>

        </div>
        """

    # ========================================================
    # PROBABILITY
    # ========================================================

    probability_html = f"""
    <div class="probability-box">

        <div class="probability-top">

            <div class="probability-label">
                Diabetes Probability
            </div>

            <strong class="probability-value">
                {probability:.2f}%
            </strong>

        </div>

        <div class="progress-background">

            <div
                class="progress-fill"
                style="width:{probability:.2f}%;">
            </div>

        </div>

        <div class="probability-info">

            Probability generated by the trained
            Logistic Regression model.

        </div>

    </div>
    """

    # ========================================================
    # SAVE PREDICTION TO SQLITE
    # ========================================================

    save_prediction(

        pregnancies=input_values[0],

        glucose=input_values[1],

        blood_pressure=input_values[2],

        skin_thickness=input_values[3],

        insulin=input_values[4],

        bmi=input_values[5],

        diabetes_pedigree=input_values[6],

        age=input_values[7],

        prediction=prediction_text,

        probability=probability

    )

    # ========================================================
    # RETURN
    # ========================================================

    return (

        result_html,

        probability_html,

        get_history()

    )


# ============================================================
# CLEAR INPUTS + RESULT ONLY
# ============================================================

def clear_inputs():

    initial_result, initial_probability = default_result()

    return (

        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",

        initial_result,

        initial_probability

    )


# ============================================================
# HEADER
# ============================================================

header = """
<div class="top-header">

    <div class="brand-section">

        <div class="logo-box">

            <svg
                class="brand-logo"
                viewBox="0 0 64 64"
                xmlns="http://www.w3.org/2000/svg"
                aria-label="Diabetes Prediction System">

                <rect
                    x="2"
                    y="2"
                    width="60"
                    height="60"
                    rx="16"
                    fill="#162b46"/>

                <path
                    d="M47 10 V22 M41 16 H53"
                    stroke="#ffffff"
                    stroke-width="3.5"
                    stroke-linecap="round"/>

                <path
                    d="M9 35
                       H18
                       L23 26
                       L29 44
                       L35 30
                       L39 35
                       H55"
                    fill="none"
                    stroke="#ffffff"
                    stroke-width="3.5"
                    stroke-linecap="round"
                    stroke-linejoin="round"/>

            </svg>

        </div>

        <div class="brand-text">

            <div class="brand-title">
                Diabetes Prediction System
            </div>

            <div class="brand-subtitle">
                MACHINE LEARNING HEALTH ANALYSIS
            </div>

        </div>

    </div>

    <div class="system-status">

        <span class="status-dot"></span>

        SYSTEM ONLINE

    </div>

</div>
"""


# ============================================================
# DISCLAIMER
# ============================================================

disclaimer = """
<div class="disclaimer">

    <div class="disclaimer-icon">
        !
    </div>

    <div class="disclaimer-text">

        <b>Educational Purpose:</b>

        This application is a machine learning project
        for educational and demonstration purposes.
        It is not a medical diagnosis tool.

    </div>

</div>
"""


# ============================================================
# CSS
# ============================================================

css = """

/* ============================================================
   GLOBAL
   ============================================================ */

* {
    box-sizing: border-box;
}

html,
body {

    margin: 0 !important;
    padding: 0 !important;

    background: #f3f6f9 !important;

    font-family:
        Inter,
        Arial,
        Helvetica,
        sans-serif !important;
}

body {
    overflow-x: hidden !important;
}

.gradio-container {

    width: 100% !important;

    max-width: 100% !important;

    min-height: 100vh !important;

    margin: 0 !important;

    padding: 0 !important;

    background: #f3f6f9 !important;
}


/* ============================================================
   REMOVE DEFAULT GRADIO STYLE
   ============================================================ */

.block,
.block.gr-box,
.gr-box,
.gr-group,
.gr-panel,
.form,
fieldset {

    border: none !important;

    box-shadow: none !important;
}

.contain {

    max-width: 100% !important;
}


/* ============================================================
   DASHBOARD
   ============================================================ */

#dashboard {

    width: 100% !important;

    max-width: 1450px !important;

    margin: 0 auto !important;

    padding: 18px 28px 16px !important;
}


/* ============================================================
   HEADER
   ============================================================ */

.top-header {

    width: 100%;

    display: flex;

    align-items: center;

    justify-content: space-between;

    padding: 3px 2px 14px;

    border-bottom: 1px solid #dce3ea;

    margin-bottom: 14px;
}

.brand-section {

    display: flex;

    align-items: center;

    gap: 11px;
}


/* ============================================================
   LOGO
   ============================================================ */

.logo-box {

    width: 46px;

    height: 46px;

    flex: 0 0 46px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 13px;

    overflow: hidden;

    background: #162b46;

    box-shadow:
        0 3px 8px rgba(22, 43, 70, 0.12);
}

.brand-logo {

    width: 46px;

    height: 46px;

    display: block;
}


/* ============================================================
   BRAND TEXT
   ============================================================ */

.brand-title {

    color: #162b46 !important;

    font-size: 21px;

    font-weight: 800;

    line-height: 1.2;
}

.brand-subtitle {

    color: #7d8997 !important;

    font-size: 8px;

    font-weight: 800;

    letter-spacing: 1.1px;

    margin-top: 3px;
}


/* ============================================================
   SYSTEM STATUS
   ============================================================ */

.system-status {

    display: flex;

    align-items: center;

    gap: 7px;

    color: #637182 !important;

    font-size: 9px;

    font-weight: 800;

    letter-spacing: 0.8px;
}

.status-dot {

    width: 7px;

    height: 7px;

    border-radius: 50%;

    background: #26a269;
}


/* ============================================================
   MAIN ROW
   ============================================================ */

#prediction-row {

    display: grid !important;

    grid-template-columns:
        minmax(0, 1.12fr)
        minmax(0, 0.88fr) !important;

    gap: 15px !important;

    width: 100% !important;

    margin: 0 !important;

    padding: 0 !important;

    align-items: stretch !important;
}


/* ============================================================
   PATIENT CARD
   ============================================================ */

#patient-inputs {

    width: 100% !important;

    min-width: 0 !important;

    background: #ffffff !important;

    border: 1px solid #dce3ea !important;

    border-radius: 14px !important;

    padding: 17px 19px 14px !important;

    box-shadow:
        0 3px 12px rgba(22, 43, 70, 0.05) !important;
}


/* ============================================================
   RESULT CARD
   ============================================================ */

#result-panel {

    width: 100% !important;

    min-width: 0 !important;

    background: #ffffff !important;

    border: 1px solid #dce3ea !important;

    border-radius: 14px !important;

    padding: 17px 19px 14px !important;

    box-shadow:
        0 3px 12px rgba(22, 43, 70, 0.05) !important;
}


/* ============================================================
   SECTION HEADINGS
   ============================================================ */

.section-number {

    color: #8793a0 !important;

    font-size: 8px;

    font-weight: 800;

    letter-spacing: 1.4px;

    margin-bottom: 3px;
}

.section-title {

    color: #162b46 !important;

    font-size: 17px;

    font-weight: 800;

    margin: 0 0 3px;
}

.section-description {

    color: #7b8794 !important;

    font-size: 10px;

    line-height: 1.4;

    margin: 0 0 11px;
}


/* ============================================================
   INPUT GRID
   ============================================================ */

#input-grid {

    display: grid !important;

    grid-template-columns:
        1fr
        1fr !important;

    gap: 6px 15px !important;

    width: 100% !important;

    margin: 0 !important;

    padding: 0 !important;
}

#input-grid > .column,
#input-grid .column {

    background: transparent !important;

    border: none !important;

    box-shadow: none !important;

    padding: 0 !important;

    margin: 0 !important;

    min-width: 0 !important;
}


/* ============================================================
   INPUT LABELS
   ============================================================ */

#patient-inputs label {

    color: #435365 !important;

    font-size: 9px !important;

    font-weight: 800 !important;

    letter-spacing: 0.3px !important;

    margin-bottom: 2px !important;
}


/* ============================================================
   INPUT BOXES
   ============================================================ */

#patient-inputs input {

    width: 100% !important;

    height: 30px !important;

    min-height: 34px !important;

    padding: 5px 9px !important;

    color: #162b46 !important;

    background: #f7f9fb !important;

    border: 1px solid #d6dfe7 !important;

    border-radius: 7px !important;

    font-size: 10.5px !important;

    box-shadow: none !important;
}

#patient-inputs input:hover {

    border-color: #b7c4d0 !important;
}

#patient-inputs input:focus {

    border-color: #59758f !important;

    box-shadow:
        0 0 0 2px rgba(89,117,143,0.10) !important;
}

#patient-inputs input::placeholder {

    color: #a1acb7 !important;
}


/* ============================================================
   BUTTON ROW
   ============================================================ */

#button-row {

    display: grid !important;

    grid-template-columns:
        1fr
        85px !important;

    gap: 8px !important;

    width: 100% !important;

    margin-top: 10px !important;
}


/* ============================================================
   PREDICTION BUTTON
   ============================================================ */

#predict-button {

    width: 100% !important;

    height: 37px !important;

    min-height: 37px !important;

    background: #162b46 !important;

    color: #ffffff !important;

    border: none !important;

    border-radius: 7px !important;

    font-size: 10.5px !important;

    font-weight: 800 !important;

    letter-spacing: 0.15px;

    box-shadow: none !important;
}

#predict-button:hover {

    background: #294663 !important;
}


/* ============================================================
   CLEAR BUTTON
   ============================================================ */

#clear-button {

    width: 100% !important;

    height: 37px !important;

    min-height: 37px !important;

    margin: 0 !important;

    background: #ffffff !important;

    color: #687786 !important;

    border: 1px solid #d6dfe7 !important;

    border-radius: 7px !important;

    font-size: 9px !important;

    font-weight: 700 !important;
}

#clear-button:hover {

    background: #f7f9fb !important;

    color: #162b46 !important;
}


/* ============================================================
   LOCAL MODEL NOTE
   ============================================================ */

.model-note {

    margin-top: 7px;

    color: #8995a2 !important;

    font-size: 8px;

    text-align: center;
}


/* ============================================================
   RESULT
   ============================================================ */

.result-content {

    min-height: 198px;

    width: 100%;

    border-radius: 11px;

    padding: 20px;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    text-align: center;
}

.neutral-result {

    background: #f7f9fb !important;

    border: 1px solid #e0e6ec !important;
}

.diabetes-result {

    background: #fff5f5 !important;

    border: 1px solid #f0cccc !important;
}

.healthy-result {

    background: #f3faf6 !important;

    border: 1px solid #cae7d6 !important;
}

.warning-result {

    background: #fffaf0 !important;

    border: 1px solid #efdfb9 !important;
}


/* ============================================================
   RESULT ICON
   ============================================================ */

.result-icon {

    width: 42px;

    height: 42px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 50%;

    background: #e8edf2;

    color: #52687d !important;

    font-size: 20px;

    font-weight: 800;

    margin-bottom: 8px;
}


/* ============================================================
   RESULT TEXT
   ============================================================ */

.result-status {

    color: #7e8995 !important;

    font-size: 8px;

    font-weight: 800;

    letter-spacing: 1.3px;
}

.result-title {

    color: #162b46 !important;

    font-size: 22px;

    font-weight: 800;

    margin-top: 5px;
}

.result-description {

    color: #697786 !important;

    font-size: 10px;

    line-height: 1.5;

    max-width: 350px;

    margin-top: 6px;
}

.result-description b {

    color: #162b46 !important;
}


/* ============================================================
   PROBABILITY
   ============================================================ */

.probability-box {

    width: 100%;

    margin-top: 10px;

    background: #f7f9fb !important;

    border: 1px solid #e0e6ec !important;

    border-radius: 9px;

    padding: 11px 12px;
}

.probability-top {

    display: flex !important;

    justify-content: space-between !important;

    align-items: center !important;

    width: 100% !important;

    background: transparent !important;

    opacity: 1 !important;

    visibility: visible !important;
}


/* ============================================================
   DIABETES PROBABILITY LABEL
   ============================================================ */

.probability-label {

    display: block !important;

    color: #162b46 !important;

    background: transparent !important;

    font-size: 10px !important;

    font-weight: 800 !important;

    line-height: 1.4 !important;

    opacity: 1 !important;

    visibility: visible !important;

    text-shadow: none !important;
}


/* ============================================================
   PROBABILITY VALUE
   ============================================================ */

.probability-value {

    display: block !important;

    color: #162b46 !important;

    background: transparent !important;

    font-size: 16px !important;

    font-weight: 800 !important;

    line-height: 1.4 !important;

    opacity: 1 !important;

    visibility: visible !important;

    text-shadow: none !important;
}


/* ============================================================
   PROGRESS BAR
   ============================================================ */

.progress-background {

    width: 100%;

    height: 6px;

    background: #dfe5ea !important;

    border-radius: 20px;

    margin-top: 7px;

    overflow: hidden;
}

.progress-fill {

    height: 100%;

    background: #597991 !important;

    border-radius: 20px;
}


/* ============================================================
   PROBABILITY INFORMATION
   ============================================================ */

.probability-info {

    color: #687786 !important;

    font-size: 8.5px !important;

    line-height: 1.4;

    margin-top: 6px;

    opacity: 1 !important;

    visibility: visible !important;
}


/* ============================================================
   HISTORY
   ============================================================ */

#history-panel {

    width: 100% !important;

    background: #ffffff !important;

    border: 1px solid #dce3ea !important;

    border-radius: 14px !important;

    padding: 11px 15px !important;

    margin-top: 12px !important;

    box-shadow:
        0 3px 12px rgba(22, 43, 70, 0.04) !important;
}

#history-panel .section-title {

    font-size: 12px !important;
}

#history-panel .section-description {

    font-size: 8.5px !important;

    margin-bottom: 3px !important;
}

.history-table-container {

    width: 100%;

    overflow-x: auto;

    margin-top: 2px;
}

.history-table {

    width: 100%;

    border-collapse: collapse;

    font-size: 8.5px;
}

.history-table th {

    text-align: left;

    color: #6b7887 !important;

    background: #f7f9fb !important;

    padding: 6px 7px;

    border-bottom: 1px solid #e1e6eb;

    font-size: 8px;
}

.history-table td {

    color: #425365 !important;

    padding: 6px 7px;

    border-bottom: 1px solid #edf0f3;
}

.history-table tr:last-child td {

    border-bottom: none;
}


/* ============================================================
   BADGES
   ============================================================ */

.badge {

    display: inline-block;

    padding: 3px 6px;

    border-radius: 20px;

    font-size: 7.5px;

    font-weight: 800;
}

.diabetes-badge {

    background: #fde4e4 !important;

    color: #b33a3a !important;
}

.healthy-badge {

    background: #e2f4e9 !important;

    color: #28794b !important;
}


/* ============================================================
   EMPTY HISTORY
   ============================================================ */

.empty-history {

    text-align: center;

    padding: 12px 8px;
}

.empty-icon {

    font-size: 19px;

    color: #8996a3 !important;
}

.empty-title {

    color: #526273 !important;

    font-size: 9px;

    font-weight: 700;

    margin-top: 4px;
}

.empty-text {

    color: #8a96a3 !important;

    font-size: 8px;

    margin-top: 2px;
}


/* ============================================================
   DISCLAIMER
   ============================================================ */

.disclaimer {

    width: 100%;

    display: flex;

    align-items: flex-start;

    gap: 8px;

    margin-top: 9px;

    padding: 8px 10px;

    background: #fff9ef !important;

    border: 1px solid #efdfc1 !important;

    border-radius: 8px;

    color: #3f4b59 !important;

    font-size: 8px;

    font-weight: 600;

    line-height: 1.5;
}

.disclaimer-text {

    color: #3f4b59 !important;

    opacity: 1 !important;
}

.disclaimer-icon {

    width: 17px;

    height: 17px;

    flex: 0 0 17px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 50%;

    background: #f4dfbb !important;

    color: #6f532f !important;

    font-size: 10px;

    font-weight: 800;
}

.disclaimer b {

    color: #162b46 !important;

    font-weight: 800;
}


/* ============================================================
   EXTRA DISCLAIMER OVERRIDE
   ============================================================ */

.disclaimer,
.disclaimer *,
#disclaimer,
#disclaimer * {

    color: #3f4b59 !important;
}

.disclaimer b,
#disclaimer b {

    color: #162b46 !important;
}


/* ============================================================
   HIDE GRADIO FOOTER
   ============================================================ */

footer {

    display: none !important;
}

.api {

    display: none !important;
}


/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 950px) {

    #dashboard {

        padding: 16px !important;
    }

    #prediction-row {

        grid-template-columns: 1fr !important;
    }

}


@media (max-width: 600px) {

    #dashboard {

        padding: 12px !important;
    }

    .top-header {

        align-items: flex-start;
    }

    .system-status {

        font-size: 7px;
    }

    .brand-title {

        font-size: 17px;
    }

    #input-grid {

        grid-template-columns: 1fr !important;
    }

    #button-row {

        grid-template-columns: 1fr !important;
    }

}


/* ============================================================
   SHORT SCREEN
   ============================================================ */

@media (max-height: 750px) and (min-width: 951px) {

    #dashboard {

        padding-top: 10px !important;

        padding-bottom: 8px !important;
    }

    .top-header {

        padding-bottom: 8px;

        margin-bottom: 8px;
    }

    #patient-inputs,
    #result-panel {

        padding-top: 12px !important;

        padding-bottom: 10px !important;
    }

    .section-description {

        margin-bottom: 7px !important;
    }

    #input-grid {

        gap: 4px 13px !important;
    }

    #patient-inputs input {

        height: 30px !important;

        min-height: 30px !important;
    }

    #button-row {

        margin-top: 6px !important;
    }

    #predict-button,
    #clear-button {

        height: 33px !important;

        min-height: 33px !important;
    }

    .result-content {

        min-height: 165px;
    }

    .probability-box {

        margin-top: 6px;

        padding: 8px 10px;
    }

    #history-panel {

        margin-top: 7px !important;

        padding: 8px 12px !important;
    }

    .history-table th,
    .history-table td {

        padding: 4px 6px;
    }

    .disclaimer {

        margin-top: 6px;

        padding: 6px 9px;
    }

}

"""


# ============================================================
# INITIAL RESULT
# ============================================================

initial_result, initial_probability = default_result()


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="Diabetes Prediction System",
    css=css,
    theme=gr.themes.Base()
) as demo:

    # ========================================================
    # DASHBOARD
    # ========================================================

    with gr.Column(
        elem_id="dashboard"
    ):

        # ====================================================
        # HEADER
        # ====================================================

        gr.HTML(header)

        # ====================================================
        # PATIENT + PREDICTION
        # ====================================================

        with gr.Row(
            elem_id="prediction-row",
            equal_height=False
        ):

            # =================================================
            # PATIENT INFORMATION
            # =================================================

            with gr.Column(
                elem_id="patient-inputs",
                scale=1
            ):

                gr.HTML("""
                <div class="section-number">
                    01 / PATIENT
                </div>

                <div class="section-title">
                    Patient Information
                </div>

                <div class="section-description">
                    Enter the patient's health information.
                </div>
                """)

                # =============================================
                # INPUTS
                # =============================================

                with gr.Row(
                    elem_id="input-grid",
                    equal_height=False
                ):

                    with gr.Column():

                        pregnancies = gr.Textbox(
                            label="PREGNANCIES",
                            placeholder="Enter number",
                            elem_id="input-pregnancies"
                        )

                        glucose = gr.Textbox(
                            label="GLUCOSE",
                            placeholder="Enter glucose",
                            elem_id="input-glucose"
                        )

                        blood_pressure = gr.Textbox(
                            label="BLOOD PRESSURE",
                            placeholder="Enter pressure",
                            elem_id="input-bp"
                        )

                        skin_thickness = gr.Textbox(
                            label="SKIN THICKNESS",
                            placeholder="Enter thickness",
                            elem_id="input-skin"
                        )

                    with gr.Column():

                        insulin = gr.Textbox(
                            label="INSULIN",
                            placeholder="Enter insulin",
                            elem_id="input-insulin"
                        )

                        bmi = gr.Textbox(
                            label="BMI",
                            placeholder="Enter BMI",
                            elem_id="input-bmi"
                        )

                        diabetes_pedigree = gr.Textbox(
                            label="DIABETES PEDIGREE",
                            placeholder="Enter value",
                            elem_id="input-dpf"
                        )

                        age = gr.Textbox(
                            label="AGE",
                            placeholder="Enter age",
                            elem_id="input-age"
                        )

                # =============================================
                # BUTTONS
                # =============================================

                with gr.Row(
                    elem_id="button-row",
                    equal_height=True
                ):

                    predict_button = gr.Button(
                        "Predict Diabetes",
                        elem_id="predict-button"
                    )

                    clear_button = gr.Button(
                        "Clear",
                        elem_id="clear-button"
                    )

                gr.HTML("""
                <div class="model-note">
                    ● Local model processing • Predictions saved locally
                </div>
                """)

            # =================================================
            # PREDICTION SYSTEM
            # =================================================

            with gr.Column(
                elem_id="result-panel",
                scale=1
            ):

                gr.HTML("""
                <div class="section-number">
                    02 / PREDICTION
                </div>

                <div class="section-title">
                    Prediction System
                </div>

                <div class="section-description">
                    Machine learning analysis of patient information.
                </div>
                """)

                result_output = gr.HTML(
                    value=initial_result
                )

                probability_output = gr.HTML(
                    value=initial_probability
                )

        # ====================================================
        # HISTORY
        # ====================================================

        with gr.Column(
            elem_id="history-panel"
        ):

            gr.HTML("""
            <div class="section-title">
                Prediction History
            </div>

            <div class="section-description">
                Previous predictions saved locally.
            </div>
            """)

            history_output = gr.HTML(
                value=get_history()
            )

        # ====================================================
        # DISCLAIMER
        # ====================================================

        gr.HTML(
            disclaimer,
            elem_id="disclaimer"
        )

        # ====================================================
        # PREDICTION EVENT
        # ====================================================

        predict_button.click(

            fn=predict_diabetes,

            inputs=[

                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age

            ],

            outputs=[

                result_output,
                probability_output,
                history_output

            ]

        )

        # ====================================================
        # CLEAR EVENT
        # ONLY CLEARS INPUTS + RESULT
        # ====================================================

        clear_button.click(

            fn=clear_inputs,

            inputs=[],

            outputs=[

                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age,

                result_output,
                probability_output

            ]

        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    demo.launch(
        inbrowser=True,
        show_error=True
    )