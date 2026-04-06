# CV Analyzer and Scorer

A full-stack application that analyzes resumes (CVs) and generates a suitability score using machine learning. Upload a PDF CV and instantly receive a score out of 100 along with feedback and suggestions.

## Live Demo
🔗 [cv-analyzer-sepia.vercel.app](https://cv-analyzer-sepia.vercel.app)

---

## Project Description

This project provides a comprehensive solution for analyzing resumes by extracting key information such as skills, experience, and industry, then uses a machine learning model to generate a suitability score. The process involves converting PDF CVs to text, extracting relevant features using regular expressions, and feeding these features into a pre-trained XGBoost Regressor model for scoring.

---

## Features

- **PDF to Text Conversion** — Converts CVs from PDF format to plain text using PyMuPDF for easier processing
- **Skill Extraction** — Identifies and extracts relevant skills from the CV text using predefined keywords
- **Experience Extraction** — Extracts the total years of experience from the CV text
- **Industry Classification** — Categorizes the candidate's industry based on keywords found in the CV
- **ML Scoring** — Uses an XGBoost Regressor model to predict a score out of 100 based on extracted skills, experience, and industry
- **Rule-Based Feedback** — Returns human-readable feedback and suggestions based on the score tier (weak / decent / strong)
- **REST API** — FastAPI backend exposes a `/analyze-cv` endpoint that accepts PDF uploads and returns JSON results
- **Web Frontend** — Clean upload interface built with HTML/CSS where users can drop a PDF and view their results instantly

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost Regressor |
| Feature Engineering | scikit-learn (OneHotEncoder, MultiLabelBinarizer) |
| PDF Parsing | PyMuPDF (fitz) |
| Backend API | FastAPI + Uvicorn |
| Frontend | HTML / CSS / JS — designed with Claude (Anthropic) |
| Backend Deployment | Render |
| Frontend Deployment | Vercel |

---

## How It Works

1. User uploads a PDF CV via the frontend
2. FastAPI receives the file and passes it to the extraction pipeline
3. Skills, experience (years), and industry are extracted using regex
4. Features are encoded and fed into the trained XGBoost model
5. A score out of 100 is returned alongside rule-based feedback
6. Results are displayed on the frontend

---

## Dependencies

```
fastapi
uvicorn
pymupdf
pandas
numpy
xgboost
scikit-learn
python-multipart
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the backend
uvicorn main:app --reload
```

Then open `index.html` in your browser. By default the frontend points to `http://localhost:8000`.

---

## Notes

- The model is trained on a small sample dataset for demonstration purposes
- Industry detection covers Tech, Finance, and Healthcare — defaults to General if undetected
- Built as a portfolio project to demonstrate ML, API development, and full-stack deployment



