# 🎓 Placement Predictor AI

A **Flask + Machine Learning web application** that predicts a student's placement probability based on academic and skill factors.  
The project also includes a **Resume ATS Checker** that analyzes uploaded resume PDFs, extracts skills, calculates ATS score, and highlights missing skills.

This project is built with a **modern UI**, stores records using **SQLite**, provides **smart improvement suggestions**, and includes **Explainable AI (feature importance)** to show *why the prediction was made*.

---

## 🚀 Features

### ✅ Placement Prediction (ML)
- Predicts placement probability using a trained ML model
- Outputs:
  - Placement Probability (%)
  - Prediction Result (Placed ✅ / Not Placed ❌)
- Stores every prediction in database (SQLite)

### ✅ Smart Improvement Suggestions
- Suggests what to improve based on weak areas:
  - CGPA target improvement
  - Aptitude score improvement
  - Skills score improvement
  - Projects & internships recommendation
- Includes target probability goal (70%, 85%, 95%)

### ✅ Explainable AI ("Why this prediction?")
- Displays top feature importance factors
- Shows a **feature importance graph**
- Helps users understand which inputs influence results most

### ✅ Resume Upload + ATS Score Checker
- Upload Resume PDF
- Extract text from PDF
- Detects skills (Python, Java, SQL, ML, etc.)
- Provides:
  - ATS Score (0–100)
  - Detected Skills
  - Missing Skills
- Stores resume analysis history in DB

### ✅ Prediction + Resume Record Tracking
- View placement prediction history (`/records`)
- View resume analysis history (`/resume-records`)

### ✅ Modern UI
- Responsive, clean UI
- Icons + progress bars
- Attractive buttons & animations

---

## 🧠 Tech Stack

- **Frontend:** HTML, CSS, Font Awesome Icons
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn model (Joblib)
- **Database:** SQLite3
- **Resume Parsing:** PyPDF2
- **Explainability Graph:** Matplotlib

---



