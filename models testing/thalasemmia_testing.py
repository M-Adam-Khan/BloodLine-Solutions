import pandas as pd
import numpy as np
import joblib

# Load the model for Thalassemia detection
model = joblib.load("Thalasemmia Detection Model/thal_model.pkl")

# Path to the extracted CSV report
csv_path = "extracted_reports\Zain_cbc_report extracted.csv"

# Read the CSV file
df = pd.read_csv(csv_path)

# 🔹 Function to Extract Values with Improved "None" Handling
def extract_value(parameter_name):
    value = df[df['Parameter'] == parameter_name]['Value'].values
    if value.size > 0:
        str_value = str(value[0]).strip().lower()
        if str_value == "none" or str_value == "nan":  
            return 0  # Convert "None"/"NaN" to 0
        try:
            return float(value[0])  # Convert to float
        except ValueError:
            return 0  # Return 0 if conversion fails
    return 0  # Return 0 if missing

# Extract required parameters
age = extract_value('Age')
hb = extract_value('Hemoglobin (Hb)')
mch = extract_value('MCH (Mean Corpuscular Hemoglobin)')
mchc = extract_value('MCHC (Mean Corpuscular Hemoglobin Concentration)')
rdw = extract_value('RDW (Red Cell Distribution Width)')  # 🔹 Fixed handling for "None"
rbc_count = extract_value('RBC Count')

# Print extracted parameters before testing
print("\n--- 📊 Extracted Input Parameters ---")
print(f"Age: {age}")
print(f"Hemoglobin (Hb): {hb}")
print(f"MCH (Mean Corpuscular Hemoglobin): {mch}")
print(f"MCHC (Mean Corpuscular Hemoglobin Concentration): {mchc}")
print(f"RDW (Red Cell Distribution Width): {rdw}")  # 🔹 Should now be 0 instead of NaN
print(f"RBC Count: {rbc_count}")
print("--------------------------------------\n")

# Prepare the new data array
new_data = np.array([[age, hb, mch, mchc, rdw, rbc_count]])

# Define column names for the Thalassemia model
columns = ['Age', 'Hb', 'MCH', 'MCHC', 'RDW', 'RBC count']
new_data_df = pd.DataFrame(new_data, columns=columns)

# Ensure no NaN values before prediction
new_data_df = new_data_df.fillna(0).astype(float)

# Predict using the model
predictions = model.predict(new_data_df)

# Print the prediction results
for i, pred in enumerate(predictions):
    if pred == 1:
        print(f"\033[91mCase {i+1}:  Detected Thalassemia\033[0m")
    else:
        print(f"\033[92mCase {i+1}: Not Detected Thalassemia\033[0m")
