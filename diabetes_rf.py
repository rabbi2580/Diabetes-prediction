import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("diabetes.csv")
print(df.head())
print(df.shape)
print(df.info())

zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

random_model=RandomForestClassifier(n_estimators=100, random_state=42 ,n_jobs=-1, max_depth=5)
random_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('classifier', random_model)  
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

random_model_pipeline.fit(X_train, y_train)
y_pred = random_model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")  
print(conf_matrix)

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(random_model_pipeline, f)
print("Model saved as random_forest_model.pkl")