import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the heart disease prediction model
model = load_model("Heart Disease Detection Model/heart_disease.h5")

# Load the scaler (assuming it's saved as a .pkl file)
scaler = joblib.load("Heart Disease Detection Model/heart_disease_scaler.pkl")

# Path to the CSV file containing the extracted report
csv_path = "extracted_reports\Zain_cbc_report extracted.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_path)

# Step 2: Extract relevant values for model prediction (Heart disease parameters)
def extract_value(parameter_name):
    value = df[df['Parameter'] == parameter_name]['Value'].values
    return value[0] if value.size > 0 else np.nan

# Extract values from the CSV
age = extract_value('Age')
sex = extract_value('Sex')
sex = 0 if sex == 'Male' else 1  # Convert sex to 0 for Male, 1 for Female

trestbps = extract_value('Trestbps (Blood Pressure)')
chol = extract_value('Chol (Cholesterol)')
fbs = extract_value('Fbs (Fasting Blood Sugar)')
restecg = extract_value('Restecg (Resting Electrocardiographic Results)')
thalach = extract_value('Thalach (Max Heart Rate)')
exang = extract_value('Exang (Exercise Induced Angina)')
oldpeak = extract_value('Oldpeak (Depression Induced by Exercise)')

# Extracting chest pain types (cp_1, cp_2, cp_3)
cp_1 = extract_value('CP_1 (Chest Pain Type 1)')
cp_2 = extract_value('CP_2 (Chest Pain Type 2)')
cp_3 = extract_value('CP_3 (Chest Pain Type 3)')

# Extracting thalassemia values (thal_2, thal_3, thal_4)
thal_2 = extract_value('Thal_2')
thal_3 = extract_value('Thal_3')
thal_4 = extract_value('Thal_4')

# Extracting slope values (slope_1, slope_2)
slope_1 = extract_value('Slope_1 (Slope of Peak Exercise ST Segment)')
slope_2 = extract_value('Slope_2')

# Step 3: Handle missing or invalid values (replace with np.nan)
# Example: If Cholesterol is 0, replace with np.nan
if chol == 0:
    chol = np.nan

# Assuming new_data contains all the necessary 18 features for a single prediction.
new_data = np.array([[age, sex, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, cp_1, cp_2, cp_3, thal_2, thal_3, slope_1, slope_2]])

# Reshape new_data to 1 row and 16 columns (without the missing 'ca' and 'thal_1')
new_data = new_data.reshape(1, -1)

# Ensure you have the correct column names (18 columns)
columns = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'ca', 'cp_1', 'cp_2', 'cp_3', 'thal_1', 'thal_2', 'thal_3', 'slope_1', 'slope_2']

# Add missing columns that were part of model training but are missing in new data (e.g., 'ca' and 'thal_1')
# Make sure the new_data array has 18 columns by adding the missing values
new_data = np.append(new_data, np.nan)  # Add NaN for 'ca' column
new_data = np.append(new_data, np.nan)  # Add NaN for 'thal_1' column

# Reshape again to ensure the correct number of columns
new_data = new_data.reshape(1, -1)

# Create DataFrame with reshaped new_data and the correct columns
new_data_df = pd.DataFrame(new_data, columns=columns)

# Check the shape of the DataFrame to verify everything is correct
print(new_data_df.shape)
print(new_data_df.head())

# Now, proceed with scaling and predictions as usual
# Step 6: Scale the data using the scaler
new_data_scaled = scaler.transform(new_data_df)

# Step 7: Predict using the model
predictions = model.predict(new_data_scaled)

# Step 8: Output the result
for i, pred in enumerate(predictions):
    if pred == 1:
        print(f"\033[91mCase {i+1}: Detected Heart Disease\033[0m")
    else:
        print(f"\033[92mCase {i+1}: Not Detected Heart Disease\033[0m")
