import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

######
data = {
    'Age': [40, 25, 32, 48],
    'Gender': ["Male", "Female", "Female", "Male"],
    'Industry': ["Oil", "Computer", "Finance", "Mechanical"],
    'Work Experience (In Years)': [9, 2, 6, 8],
    'Language': ["Hindi", "Spanish", "English", "Portugese"],
    'Real Score': [89, 62, 85, 72]}
df = pd.DataFrame(data)
######

GenderEncoder=LabelEncoder()
IndustryEncoder=LabelEncoder()
LanguageEncoder = LabelEncoder()

df["Gender"]=GenderEncoder.fit_transform(df["Gender"])
df["Industry"]=IndustryEncoder.fit_transform(df["Industry"])
df["Language"]=LanguageEncoder.fit_transform(df["Language"])

X = df[["Age","Gender","Industry","Work Experience (In Years)","Language"]]
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


########TEST########

test = {
    'Age': [29],
    'Gender': ["Male"],
    'Industry': ["Finance"],
    'Work Experience (In Years)': [7],
    'Language': ["English"]}

test["Gender"] = GenderEncoder.transform(test["Gender"])
test["Industry"] = IndustryEncoder.transform(test["Industry"])
test["Language"] = LanguageEncoder.transform(test["Language"])



testdf=pd.DataFrame(test)

print(xgb.predict(testdf))
