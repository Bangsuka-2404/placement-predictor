from flask import Flask, render_template, request
import sqlite3
import joblib
import numpy as np
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader

# ✅ Explainability graph
import matplotlib
matplotlib.use("Agg")  # important for Flask server
import matplotlib.pyplot as plt

app = Flask(__name__)

# --------------------------
# CONFIG
# --------------------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --------------------------
# Load ML model & scaler
# --------------------------
model = joblib.load("placement_model.pkl")
scaler = joblib.load("scaler.pkl")


# --------------------------
# Database init
# --------------------------
def init_db():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    # Placement predictions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS placements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cgpa REAL,
            projects INTEGER,
            skills_score INTEGER,
            aptitude_score INTEGER,
            internships INTEGER,
            probability REAL,
            prediction TEXT
        )
    """)

    # Resume analysis table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS resume_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            ats_score REAL,
            detected_skills TEXT,
            missing_skills TEXT
        )
    """)

    conn.commit()
    conn.close()


# --------------------------
# Helper functions
# --------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.lower()


def analyze_resume(text):
    skill_keywords = [
        "python", "java", "c++", "c", "javascript", "html", "css", "react", "node",
        "flask", "django", "sql", "mysql", "mongodb", "postgresql",
        "machine learning", "deep learning", "data science", "pandas", "numpy",
        "scikit-learn", "tensorflow", "keras",
        "git", "github", "docker", "linux", "api", "rest", "cloud"
    ]

    detected = []
    for skill in skill_keywords:
        if skill in text:
            detected.append(skill)

    missing = [s for s in skill_keywords if s not in detected]
    ats_score = (len(detected) / len(skill_keywords)) * 100

    return round(ats_score, 2), detected, missing


def generate_suggestions(cgpa, projects, skills_score, aptitude_score, internships, probability):
    suggestions = []

    if probability < 50:
        target_prob = "70%"
        cgpa_target = 7.5
        skills_target = 70
        aptitude_target = 70
        projects_target = 3
        internships_target = 1

    elif probability < 75:
        target_prob = "85%"
        cgpa_target = 8.0
        skills_target = 80
        aptitude_target = 80
        projects_target = 4
        internships_target = 2

    else:
        target_prob = "95%"
        cgpa_target = 8.5
        skills_target = 90
        aptitude_target = 90
        projects_target = 5
        internships_target = 2

    if cgpa < cgpa_target:
        suggestions.append(f"📌 Increase CGPA from {cgpa} → {cgpa_target} to improve placement chances.")

    if projects < projects_target:
        suggestions.append(f"📌 Build more projects: {projects} → {projects_target}. Add projects to GitHub & LinkedIn.")

    if skills_score < skills_target:
        suggestions.append(f"📌 Improve technical skills score: {skills_score} → {skills_target}. Focus on DSA + core skills.")

    if aptitude_score < aptitude_target:
        suggestions.append(f"📌 Increase aptitude score: {aptitude_score} → {aptitude_target}. Practice aptitude daily.")

    if internships < internships_target:
        suggestions.append(f"📌 Gain internships: {internships} → {internships_target}. Apply on Internshala / LinkedIn.")

    if len(suggestions) == 0:
        suggestions.append("✅ You are doing great! Maintain consistency and focus on interviews + resume improvement.")

    return target_prob, suggestions


# --------------------------
# ✅ Explainable AI functions
# --------------------------
def get_feature_importance(model):
    """
    Returns list of (feature_name, score) sorted descending.
    Supports tree models and logistic regression.
    """
    feature_names = ["CGPA", "Projects", "Skills Score", "Aptitude Score", "Internships"]

    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_

    elif hasattr(model, "coef_"):
        scores = np.abs(model.coef_[0])

    else:
        scores = np.ones(len(feature_names))

    importance = list(zip(feature_names, scores))
    importance.sort(key=lambda x: x[1], reverse=True)

    total = sum([x[1] for x in importance])
    importance = [(name, float(score / total)) for name, score in importance]

    return importance


def save_feature_importance_graph(importance):
    """
    Saves a bar chart in static folder and returns filename.
    """
    names = [x[0] for x in importance]
    scores = [x[1] for x in importance]

    plt.figure(figsize=(7, 4))
    plt.bar(names, scores)
    plt.title("Feature Importance (Why this prediction?)")
    plt.ylabel("Importance")
    plt.xticks(rotation=20)

    filepath = os.path.join("static", "feature_importance.png")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return "feature_importance.png"


# --------------------------
# ROUTES
# --------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        cgpa = float(request.form["cgpa"])
        projects = int(request.form["projects"])
        skills_score = int(request.form["skills_score"])
        aptitude_score = int(request.form["aptitude_score"])
        internships = int(request.form["internships"])

        # ML Prediction
        features = np.array([[cgpa, projects, skills_score, aptitude_score, internships]])
        scaled_features = scaler.transform(features)

        proba = model.predict_proba(scaled_features)[0][1]
        prediction = "Placed ✅" if proba >= 0.5 else "Not Placed ❌"
        probability_percent = round(proba * 100, 2)

        # Save placement prediction
        conn = sqlite3.connect("database.db")
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO placements (cgpa, projects, skills_score, aptitude_score, internships, probability, prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (cgpa, projects, skills_score, aptitude_score, internships, float(proba), prediction))
        conn.commit()
        conn.close()

        # Smart suggestions
        target_prob, suggestions = generate_suggestions(
            cgpa, projects, skills_score, aptitude_score, internships, probability_percent
        )

        # ✅ Explainable AI
        importance = get_feature_importance(model)
        top_factors = importance[:5]
        graph_file = save_feature_importance_graph(top_factors)

        return render_template(
            "results.html",
            cgpa=cgpa,
            projects=projects,
            skills_score=skills_score,
            aptitude_score=aptitude_score,
            internships=internships,
            probability=probability_percent,
            prediction=prediction,
            target_prob=target_prob,
            suggestions=suggestions,
            top_factors=top_factors,
            graph_file=graph_file
        )

    return render_template("index.html")


@app.route("/records")
def records():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM placements ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()

    return render_template("records.html", rows=rows)


# Resume Upload + ATS
@app.route("/resume", methods=["GET", "POST"])
def resume():
    if request.method == "POST":
        if "resume" not in request.files:
            return "No file uploaded!"

        file = request.files["resume"]

        if file.filename == "":
            return "No file selected!"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Extract PDF text
            text = extract_text_from_pdf(filepath)

            # Analyze resume
            ats_score, detected_skills, missing_skills = analyze_resume(text)

            # Save resume analysis to DB
            conn = sqlite3.connect("database.db")
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO resume_analysis (filename, ats_score, detected_skills, missing_skills)
                VALUES (?, ?, ?, ?)
            """, (
                filename,
                float(ats_score),
                ", ".join(detected_skills),
                ", ".join(missing_skills)
            ))
            conn.commit()
            conn.close()

            return render_template(
                "resume_result.html",
                filename=filename,
                ats_score=ats_score,
                detected_skills=detected_skills,
                missing_skills=missing_skills
            )

        return "Only PDF files allowed!"

    return render_template("resume.html")


@app.route("/resume-records")
def resume_records():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM resume_analysis ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return render_template("resume_records.html", rows=rows)


# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
