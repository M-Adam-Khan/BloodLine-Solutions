import pandas as pd
import numpy as np
import joblib

model = joblib.load("Anemia Detection Model/anemia_model.pkl")
scaler = joblib.load("Anemia Detection Model/scaler.pkl")

csv_path = "extracted_reports/Zain_cbc_report extracted.csv"

df = pd.read_csv(csv_path)

mcv = df[df['Parameter'] == 'MCV (Mean Corpuscular Volume)']['Value'].values[0]
mchc = df[df['Parameter'] == 'MCHC (Mean Corpuscular Hemoglobin Concentration)']['Value'].values[0]
mch = df[df['Parameter'] == 'MCH (Mean Corpuscular Hemoglobin)']['Value'].values[0]
hemoglobin = df[df['Parameter'] == 'Hemoglobin (Hb)']['Value'].values[0]

sex = df[df['Parameter'] == 'Sex']['Value'].values[0]
gender = 0 if 'Male' in sex else 1  

if mcv == 0:
    mcv = np.nan  

new_data = np.array([[mcv, mchc, mch, hemoglobin, gender]])

columns = ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']
new_data_df = pd.DataFrame(new_data, columns=columns)

new_data_scaled = scaler.transform(new_data_df)

# Step 6: Predict using the model
predictions = model.predict(new_data_scaled)

for i, pred in enumerate(predictions):
    if pred == 1:
        print(f"\033[91mCase {i+1}: Detected Anemia\033[0m")
    else:
        print(f"\033[92mCase {i+1}: Not Detected Anemia\033[0m")
