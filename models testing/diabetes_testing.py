import pandas as pd
import numpy as np
import joblib

model = joblib.load("Diabetes Detection Model\diabetes_model.pkl")
scaler = joblib.load("Diabetes Detection Model\diabetes_scaler.pkl")

csv_path = "extracted_reports\Zain_cbc_report extracted.csv"

df = pd.read_csv(csv_path)

pregnancies = df[df['Parameter'] == 'Pregnancies']['Value'].values[0]
glucose = df[df['Parameter'] == 'Glucose']['Value'].values[0]
blood_pressure = df[df['Parameter'] == 'Blood Pressure']['Value'].values[0]
skin_thickness = df[df['Parameter'] == 'Skin Thickness']['Value'].values[0]
insulin = df[df['Parameter'] == 'Insulin']['Value'].values[0]
bmi = df[df['Parameter'] == 'BMI']['Value'].values[0]
diabetes_pedigree_function = df[df['Parameter'] == 'Diabetes Pedigree Function']['Value'].values[0]
age = df[df['Parameter'] == 'Age']['Value'].values[0]

if glucose == 0:
    glucose = np.nan

new_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
new_data_df = pd.DataFrame(new_data, columns=columns)

new_data_scaled = scaler.transform(new_data_df)

predictions = model.predict(new_data_scaled)

for i, pred in enumerate(predictions):
    if pred == 1:
        print(f"\033[91mCase {i+1}: Detected Diabetes\033[0m")
    else:
        print(f"\033[92mCase {i+1}: Not Detected Diabetes\033[0m")
