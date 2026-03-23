**CV Analyzer and Scorer**
Project Description
This project provides a comprehensive solution for analyzing resumes (CVs) by extracting key information such as skills, experience, and industry, and then uses a machine learning model to generate a suitability score for the candidate. The process involves converting PDF CVs to text, extracting relevant features using regular expressions, and feeding these features into a pre-trained XGBoost Regressor model for scoring.

Features
PDF to Text Conversion: Converts CVs from PDF format to plain text for easier processing.
Skill Extraction: Identifies and extracts relevant skills from the CV text using predefined keywords.
Experience Extraction: Extracts the total years of experience from the CV text.
Industry Classification: Categorizes the candidate's industry based on keywords found in the CV.
Machine Learning Scoring: Utilizes an XGBoost Regressor model to predict a 'Real Score' based on the extracted skills, experience, and industry, helping to rank candidates.
Dependencies
The project requires the following Python libraries:

PyMuPDF (fitz): For PDF parsing.
pandas: For data manipulation and DataFrame operations.
numpy: For numerical operations.
scikit-learn: For data preprocessing (OneHotEncoder, MultiLabelBinarizer) and model evaluation.
xgboost: For the machine learning regression model.
re: For regular expression operations.



