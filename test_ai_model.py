import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer 

######
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
        ["python", "ml", "java", "sql", "git"]],
    'experience': [5, 10, 2, 8, 0, 15, 1, 12, 4, 6],
    'industry': [
        "Tech", "Finance", "Tech", "Tech", "General",
        "Healthcare", "Tech", "Finance", "Finance", "Tech"
    ],
    'Real Score': [85, 92, 45, 88, 10, 98, 35, 95, 40, 90]}
df = pd.DataFrame(data)
######


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
    objective='reg:squarederror')


xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

print(y_pred)


import re



extracted_experience_list = extract_experience(cv.lower())
experience_value_for_ohe = 0
if extracted_experience_list:

    match = re.search(r'(\d+)', extracted_experience_list[0])
    if match:
        experience_value_for_ohe = int(match.group(1))

test_candidate_data = {
    'skills': [extract_skill(cv.lower())],
    'experience': [experience_value_for_ohe],
    'industry': [industry(cv.lower())]
}

new_df_for_prediction = pd.DataFrame(test_candidate_data);


new_skills_encoded = skills_mlb.transform(new_df_for_prediction["skills"])
new_skills_df = pd.DataFrame(new_skills_encoded, columns=skills_mlb.classes_, index=new_df_for_prediction.index)


new_experience_encoded = experience_ohe.transform(new_df_for_prediction[['experience']])
new_experience_df = pd.DataFrame(new_experience_encoded, columns=experience_ohe.get_feature_names_out(['experience']), index=new_df_for_prediction.index)


new_industry_encoded = industry_ohe.transform(new_df_for_prediction[['industry']])
new_industry_df = pd.DataFrame(new_industry_encoded, columns=industry_ohe.get_feature_names_out(['industry']), index=new_df_for_prediction.index)


processed_new_df_features = pd.concat([new_skills_df, new_experience_df, new_industry_df], axis=1)

final_test_features = pd.DataFrame(0, index=range(len(processed_new_df_features)), columns=X.columns)

for col in processed_new_df_features.columns:
    if col in final_test_features.columns:
        final_test_features[col] = processed_new_df_features[col]


print(xgb.predict(final_test_features))
