import os
import pandas as pd
import numpy as np
import joblib
import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model

# Suppress Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load Models & Scalers
models = {
    "Anemia": joblib.load("Anemia Detection Model/anemia_model.pkl"),
    "Diabetes": joblib.load("Diabetes Detection Model/diabetes_model.pkl"),
    "Thalassemia": joblib.load("Thalasemmia Detection Model/thal_model.pkl"),
    "Heart Disease": load_model("Heart Disease Detection Model/heart_disease.h5"),
}

scalers = {
    "Anemia": joblib.load("Anemia Detection Model/scaler.pkl"),
    "Diabetes": joblib.load("Diabetes Detection Model/diabetes_scaler.pkl"),
    "Heart Disease": joblib.load("Heart Disease Detection Model/heart_disease_scaler.pkl"),
}

# Load Extracted Report
csv_path = "extracted_reports\Ijaz_cbc_report extracted.csv"
df = pd.read_csv(csv_path)

# 🔹 Helper Function to Extract Values with Default 0 if Missing
def extract_value(parameter_name):
    value = df[df['Parameter'] == parameter_name]['Value'].values
    if value.size > 0:
        if parameter_name == "Sex":
            return 0 if str(value[0]).strip().lower() == 'male' else 1  # Male → 0, Female → 1
        if parameter_name == "Age":
            return float(value[0])  # Ensure age is an integer
        try:
            return float(value[0])  # Convert numerical values to float
        except ValueError:
            return 0  # If conversion fails, return 0
    return 0  # If missing, return 0
# Extract Age and Gender
age = extract_value('Age')
gender = extract_value('Sex') 
sex = gender
 # Now stores 0 for Male, 1 for Female


# ---- 1. ANEMIA DETECTION ----
anemia_data = np.array([[extract_value('MCV (Mean Corpuscular Volume)'),
                         extract_value('MCHC (Mean Corpuscular Hemoglobin Concentration)'),
                         extract_value('MCH (Mean Corpuscular Hemoglobin)'),
                         extract_value('Hemoglobin (Hb)'),
                         gender]])
anemia_df = pd.DataFrame(anemia_data, columns=['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV'])
anemia_df_scaled = scalers["Anemia"].transform(anemia_df)
anemia_pred = models["Anemia"].predict(anemia_df_scaled)[0]

# ---- 2. DIABETES DETECTION ----
diabetes_data = np.array([[extract_value('Pregnancies'),
                           extract_value('Glucose'),
                           extract_value('Blood Pressure'),
                           extract_value('Skin Thickness'),
                           extract_value('Insulin'),
                           extract_value('BMI'),
                           extract_value('Diabetes Pedigree Function'),
                           age]])
diabetes_df = pd.DataFrame(diabetes_data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
diabetes_df_scaled = scalers["Diabetes"].transform(diabetes_df)
diabetes_pred = models["Diabetes"].predict(diabetes_df_scaled)[0]

# ---- 3. THALASSEMIA DETECTION ----
thalassemia_data = np.array([[age,
                              extract_value('Hemoglobin (Hb)'),
                              extract_value('MCH (Mean Corpuscular Hemoglobin)'),
                              extract_value('MCHC (Mean Corpuscular Hemoglobin Concentration)'),
                              extract_value('RDW (Red Cell Distribution Width)'),
                              extract_value('RBC Count')]])

thalassemia_df = pd.DataFrame(thalassemia_data, columns=['Age', 'Hb', 'MCH', 'MCHC', 'RDW', 'RBC count'])

# 🔹 **ENSURE NO NaN VALUES** BEFORE MODEL PREDICTION
#thalassemia_df.fillna(0, inplace=True)  # Convert all NaNs to 0
#thalassemia_df = thalassemia_df.astype(float)  # Ensure float values

# 🔹 Predict Using the Model
thalassemia_pred = models["Thalassemia"].predict(thalassemia_df)[0]

# ---- 4. HEART DISEASE DETECTION ----
heart_data = np.array([[age, sex, 
                        extract_value('Trestbps (Blood Pressure)'),  
                        extract_value('Chol (Cholesterol)'),  
                        extract_value('Fbs (Fasting Blood Sugar)'),  
                        extract_value('Restecg (Resting Electrocardiographic Results)'),  
                        extract_value('Thalach (Max Heart Rate)'),  
                        extract_value('Exang (Exercise Induced Angina)'),  
                        extract_value('Oldpeak (Depression Induced by Exercise)'),  
                        extract_value('CP_1 (Chest Pain Type 1)'),  
                        extract_value('CP_2 (Chest Pain Type 2)'),  
                        extract_value('CP_3 (Chest Pain Type 3)'),  
                        extract_value('Thal_2'),  
                        extract_value('Thal_3'),  
                        extract_value('Slope_1 (Slope of Peak Exercise ST Segment)'),  
                        extract_value('Slope_2')]])

# Add missing 'ca' and 'thal_1' values as 0
heart_data = np.append(heart_data, 0)  # Add 0 for 'ca' column
heart_data = np.append(heart_data, 0)  # Add 0 for 'thal_1' column
heart_data = heart_data.reshape(1, -1)

# Create DataFrame
heart_columns = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'ca',  
                 'cp_1', 'cp_2', 'cp_3', 'thal_1', 'thal_2', 'thal_3', 'slope_1', 'slope_2']
heart_df = pd.DataFrame(heart_data, columns=heart_columns)

# Scale the data
heart_df_scaled = scalers["Heart Disease"].transform(heart_df)

# Predict using the heart disease model
heart_pred = models["Heart Disease"].predict(heart_df_scaled)[0]

# ---- Display Final Results ----
print("\n--- Prediction Results ---")
print(f"Anemia: {'Detected' if anemia_pred == 1 else 'Not Detected'}")
print(f"Diabetes: {'Detected' if diabetes_pred == 1 else 'Not Detected'}")
print(f"Thalassemia: {'Detected' if thalassemia_pred == 1 else 'Not Detected'}")
print(f"Heart Disease: {'Detected' if heart_pred == 1 else 'Not Detected'}")
