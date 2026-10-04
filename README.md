# 🩺 Diabetes Prediction System

### Machine Learning-Based Health Analysis Web Application

The **Diabetes Prediction System** is a machine learning web application developed to predict whether a person is likely to have diabetes based on selected health-related parameters.

The system uses a **Logistic Regression** machine learning model trained on the **Pima Indians Diabetes Dataset**. Users can enter patient information through a clean dashboard and receive a prediction along with the estimated probability.

> ⚠️ **Educational Disclaimer:** This application is developed for educational and demonstration purposes only. It is **not a medical diagnosis tool** and should not be used as a substitute for professional medical advice.

---

## 📌 About the Project

Diabetes is a common health condition that can be influenced by several factors such as glucose level, BMI, age, blood pressure, insulin level, and other health-related attributes.

This project demonstrates how **Machine Learning can be integrated with a web-based interface** to analyze these parameters and generate a diabetes prediction.

The application provides:

- 👤 Patient information input
- 🤖 Machine learning prediction
- 📊 Diabetes probability
- 🗄️ SQLite prediction history
- 🧹 Input clearing functionality
- 💻 Interactive web dashboard
- ⚡ Local model processing

---

## ✨ Features

### 👤 Patient Information

The system accepts the following eight health-related parameters:

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| Blood Pressure | Diastolic blood pressure |
| Skin Thickness | Triceps skin fold thickness |
| Insulin | 2-Hour serum insulin |
| BMI | Body Mass Index |
| Diabetes Pedigree Function | Diabetes-related genetic influence |
| Age | Patient age |

---

### 🤖 Machine Learning Prediction

The application uses **Logistic Regression** to classify the input into two categories:

- 🟢 **No Diabetes**
- 🔴 **Diabetes**

The system also displays the estimated probability associated with the prediction.

---

### 🗄️ Prediction History

Every prediction is stored locally using **SQLite**.

The database records:

- Glucose
- BMI
- Age
- Prediction
- Probability
- Date and time

The prediction history remains available even after clearing the current input fields.

---

### 🧹 Clear Function

The **Clear** button resets:

- Patient input fields
- Current prediction
- Probability display

It does **not** delete the stored prediction history.

---

## 🧠 Machine Learning Workflow

The machine learning pipeline follows these steps:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Train / Test Split
   ↓
Missing Value Handling
   ↓
Feature Scaling
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
Gradio Web Application
   ↓
Diabetes Prediction
```

### Data Preprocessing

The dataset contains several health-related fields where a value of `0` may represent a missing or invalid measurement.

Zeros are treated as missing values for:

```text
Glucose
BloodPressure
SkinThickness
Insulin
BMI
```

Missing values are handled using:

**Median Imputation**

The features are then standardized using:

**StandardScaler**

---

## 🤖 Machine Learning Model

### Logistic Regression

The project uses **Logistic Regression** as the classification algorithm.

Logistic Regression was selected because it is:

- Simple
- Efficient
- Suitable for binary classification
- Easy to interpret
- Appropriate for demonstrating a healthcare classification problem

The model predicts one of two outcomes:

```text
0 → No Diabetes
1 → Diabetes
```

---

## 📊 Model Performance

The model was evaluated using an 80/20 train-test split with stratification.

### Accuracy

**70.78%**

### Confusion Matrix

```text
[[82, 18],
 [27, 27]]
```

### Classification Report

```text
              precision    recall  f1-score   support

           0       0.75      0.82      0.78       100
           1       0.60      0.50      0.55        54

    accuracy                           0.71       154
   macro avg       0.68      0.66      0.67       154
weighted avg       0.70      0.71      0.70       154
```

> The model performance is intended for academic demonstration and should not be interpreted as clinical performance.

---

## 🛠️ Technologies Used

### Programming Language

- 🐍 **Python**

### Machine Learning

- **Scikit-learn**
- Logistic Regression
- SimpleImputer
- StandardScaler
- Train/Test Split

### Data Processing

- **Pandas**
- **NumPy**

### Web Interface

- **Gradio**
- HTML
- CSS

### Database

- **SQLite**

### Model Storage

- **Joblib**

### Development Environment

- **Visual Studio Code**
- **Git**
- **GitHub**

---

## 📂 Project Structure

```text
Diabetes-Prediction-System/
│
├── app.py
│       └── Main Gradio web application
│
├── analysis.py
│       └── Data analysis and model development
│
├── database.py
│       └── SQLite database operations
│
├── diabetes.csv
│       └── Pima Indians Diabetes Dataset
│
├── diabetes_model.pkl
│       └── Trained Logistic Regression model
│
├── scaler.pkl
│       └── Feature scaling model
│
├── imputer.pkl
│       └── Missing-value imputation model
│
├── .gitignore
│       └── Git ignored files
│
└── README.md
        └── Project documentation
```

### Local Database

The file:

```text
predictions.db
```

is created automatically when the application runs.

It is intentionally excluded from GitHub through `.gitignore` because it contains locally generated prediction history.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Monika-M-19/Diabetes-Prediction-System.git
```

### 2. Navigate to the Project

```bash
cd Diabetes-Prediction-System
```

### 3. Install Required Libraries

```bash
pip install pandas numpy scikit-learn joblib gradio
```

SQLite is included with Python, so no separate SQLite installation is required.

---

## ▶️ How to Run

Run the following command from the project directory:

```bash
python app.py
```

The Gradio application will start locally.

The application will open in your browser, where you can enter the patient information and generate a prediction.

---

## 🧪 Example Test Data

The following values can be used for **demonstration/testing only**.

### Example 1 — Higher-Risk Test Input

```text
Pregnancies: 6
Glucose: 180
Blood Pressure: 90
Skin Thickness: 35
Insulin: 200
BMI: 35
Diabetes Pedigree Function: 0.80
Age: 50
```

### Example 2 — Lower-Risk Test Input

```text
Pregnancies: 1
Glucose: 90
Blood Pressure: 62
Skin Thickness: 12
Insulin: 43
BMI: 27.2
Diabetes Pedigree Function: 0.58
Age: 24
```

> These examples are provided only to demonstrate application functionality. The application's output should not be considered a medical diagnosis.

---

## 🔐 Data & Privacy

The application performs model processing locally.

Prediction history is stored in a local SQLite database:

```text
predictions.db
```

The database is excluded from version control using `.gitignore`.

---

## 🚀 Future Enhancements

Possible improvements for future versions include:

- 📈 Interactive prediction analytics
- 📊 More detailed visualization dashboards
- 🧠 Comparison of multiple machine learning algorithms
- 🎯 Hyperparameter tuning
- 📱 Improved mobile responsiveness
- 👤 User authentication
- 📄 Downloadable prediction reports
- ☁️ Cloud deployment
- 🔐 Enhanced data security
- 📊 Model performance monitoring
- 🧪 Cross-validation and additional evaluation metrics

---

## 🎓 Academic Purpose

This project was developed as an **academic Machine Learning project** to demonstrate the practical implementation of:

- Data preprocessing
- Missing value handling
- Feature scaling
- Binary classification
- Logistic Regression
- Model evaluation
- Model serialization
- Web application development
- Database integration

---

## 👩‍💻 Author

**Monika M**

MCA Student  
St. Agnes College (Autonomous), Mangaluru

### GitHub

[GitHub Profile](https://github.com/Monika-M-19)

---

## 📜 License

This project is intended for **educational and academic purposes**.

You may use the project for learning and demonstration purposes.

---

## ⭐ Support

If you find this project useful for learning, consider giving the repository a ⭐ on GitHub.