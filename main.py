from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz
import re
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Train model once on startup ──────────────────────────────────────────────
data = {
    'skills': [
        ["python", "sql", "machine learning"],
        ["excel", "accounting", "audit"],
        ["python", "c++"],
        ["java", "sql", "software", "cloud"],
        [],
        ["medical", "surgery", "nurse", "clinical"],
        ["python"],
        ["fintech", "equity", "investment"],
        ["excel"],
        ["python", "ml", "java", "sql", "git"]
    ],
    'experience': [5, 10, 2, 8, 0, 15, 1, 12, 4, 6],
    'industry': [
        "Tech", "Finance", "Tech", "Tech", "General",
        "Healthcare", "Tech", "Finance", "Finance", "Tech"
    ],
    'Real Score': [85, 92, 45, 88, 10, 98, 35, 95, 40, 90]
}

df = pd.DataFrame(data)

skills_mlb = MultiLabelBinarizer()
experience_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
industry_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

skills_encoded = skills_mlb.fit_transform(df["skills"])
skills_df = pd.DataFrame(skills_encoded, columns=skills_mlb.classes_, index=df.index)
df = pd.concat([df.drop("skills", axis=1), skills_df], axis=1)

experience_encoded = experience_ohe.fit_transform(df[['experience']])
experience_df = pd.DataFrame(experience_encoded, columns=experience_ohe.get_feature_names_out(['experience']), index=df.index)
df = pd.concat([df.drop("experience", axis=1), experience_df], axis=1)

industry_encoded = industry_ohe.fit_transform(df[['industry']])
industry_df = pd.DataFrame(industry_encoded, columns=industry_ohe.get_feature_names_out(['industry']), index=df.index)
df = pd.concat([df.drop("industry", axis=1), industry_df], axis=1)

X = df.drop('Real Score', axis=1)
y = df['Real Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    objective='reg:squarederror'
)
xgb.fit(X_train, y_train)

# ── Your original functions ──────────────────────────────────────────────────
def extraction_of_cv(filepath):
    doc = fitz.open(filepath)
    text = ""
    for i in doc:
        text += i.get_text()
    return text

def extract_skill(cvtest):
    skills = ["python", "sql", r"c\+\+", "machine learning", "java",
              "excel", "accounting", "audit", "fintech", "equity",
              "investment", "medical", "surgery", "nurse", "clinical",
              "cloud", "software", "git", "ml"]
    pattern = r'\b(' + '|'.join(skills) + r')\b'
    return list(set(re.findall(pattern, cvtest.lower())))

def extract_experience(exptest):
    return re.findall(r'\d+\s+years', exptest)

def industry(industrytest):
    industries = {
        "Tech": r"\b(software|python|java|cloud|developer|data science|ai|ml)\b",
        "Finance": r"\b(banking|accounting|audit|fintech|investment|equity|excel)\b",
        "Healthcare": r"\b(medical|nurse|patient|clinical|hospital|pharmacy|surgery)\b"
    }
    for ind, pattern in industries.items():
        if re.search(pattern, industrytest):
            return ind
    return "General"  # fallback — prevents crash when no industry matches

# ── Rule-based feedback ──────────────────────────────────────────────────────
def generate_feedback(score, skills, experience, detected_industry):
    feedback = []

    if score < 50:
        feedback.append("Your CV scored in the weak range. Here's what to work on:")
    elif score < 75:
        feedback.append("Your CV scored in the decent range. A few improvements could push it higher:")
    else:
        feedback.append("Strong CV! Here are some tips to keep it polished:")

    if len(skills) == 0:
        feedback.append("• No recognisable skills were detected. Make sure your skills are listed clearly — e.g. Python, SQL, Java.")
    elif len(skills) < 3:
        feedback.append(f"• Only {len(skills)} skill(s) detected ({', '.join(skills)}). Consider adding more relevant technical or domain skills.")
    else:
        feedback.append(f"• {len(skills)} skills detected: {', '.join(skills)}. Good range of skills.")

    if experience == 0:
        feedback.append("• No years of experience found. Make sure to mention your experience clearly, e.g. '3 years of experience in...'")
    elif experience < 3:
        feedback.append(f"• {experience} year(s) of experience detected. Early career is fine — highlight projects and achievements to compensate.")
    else:
        feedback.append(f"• {experience} years of experience detected. Strong experience level.")

    if detected_industry == "General":
        feedback.append("• Your industry couldn't be identified. Try including clearer industry keywords relevant to your field.")
    else:
        feedback.append(f"• Industry identified as: {detected_industry}.")

    if score < 50:
        feedback.append("• Overall: Focus on clearly listing your skills, quantifying your experience, and using industry-relevant keywords.")
    elif score < 75:
        feedback.append("• Overall: You're on the right track. Expanding your skill set and clarifying your experience will boost your score.")
    else:
        feedback.append("• Overall: Great foundation. Make sure your CV is tailored to each role you apply for.")

    return "\n".join(feedback)

# ── The endpoint ─────────────────────────────────────────────────────────────
@app.post("/analyze-cv")
async def analyze_cv(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        cv = extraction_of_cv(tmp_path)
    finally:
        os.remove(tmp_path)

    extracted_experience_list = extract_experience(cv.lower())
    experience_value_for_ohe = 0
    if extracted_experience_list:
        match = re.search(r'(\d+)', extracted_experience_list[0])
        if match:
            experience_value_for_ohe = int(match.group(1))

    skills_found = extract_skill(cv.lower())
    detected_industry = industry(cv.lower())

    test_candidate_data = {
        'skills': [skills_found],
        'experience': [experience_value_for_ohe],
        'industry': [detected_industry]
    }

    new_df = pd.DataFrame(test_candidate_data)

    new_skills_encoded = skills_mlb.transform(new_df["skills"])
    new_skills_df = pd.DataFrame(new_skills_encoded, columns=skills_mlb.classes_, index=new_df.index)

    new_experience_encoded = experience_ohe.transform(new_df[['experience']])
    new_experience_df = pd.DataFrame(new_experience_encoded, columns=experience_ohe.get_feature_names_out(['experience']), index=new_df.index)

    new_industry_encoded = industry_ohe.transform(new_df[['industry']])
    new_industry_df = pd.DataFrame(new_industry_encoded, columns=industry_ohe.get_feature_names_out(['industry']), index=new_df.index)

    processed = pd.concat([new_skills_df, new_experience_df, new_industry_df], axis=1)
    final_features = pd.DataFrame(0, index=range(len(processed)), columns=X.columns)
    for col in processed.columns:
        if col in final_features.columns:
            final_features[col] = processed[col]

    score = round(float(xgb.predict(final_features)[0]), 1)
    score = max(0, min(100, score))  # clamp between 0-100

    feedback = generate_feedback(score, skills_found, experience_value_for_ohe, detected_industry)

    return {
        "score": score,
        "feedback": feedback
    }

@app.get("/")
def root():
    return {"status": "CV Analyzer API is running"}
