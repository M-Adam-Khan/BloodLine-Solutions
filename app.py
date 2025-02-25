import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from neo4j import GraphDatabase
from flask import Flask, render_template, request, redirect, url_for, jsonify
from text_extraction import process_medical_report
from tensorflow.keras.models import load_model
from chatbot import Chatbot



app = Flask(__name__)

bot = Chatbot()  

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

NEO4J_URI = "bolt://localhost:7687" 
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "donor123"

@app.route('/chatbot')  # Route for chatbot UI
def chatbot_page():
    return render_template('chatbot.html')

@app.route('/get_response', methods=['POST'])  # Chatbot response API
def get_chatbot_response():
    user_input = request.json.get("message")
    response = bot.get_response(user_input)
    return jsonify({"response": response})


class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, query, parameters={}):
     with self.driver.session() as session:
        result = session.run(query, parameters)
        return [record for record in result] 

# Initialize the database connection
db = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

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

def extract_value(df, parameter_name):
    value = df[df['Parameter'] == parameter_name]['Value'].values
    if value.size > 0:
        try:
            return float(value[0])  # Convert to float
        except ValueError:
            return 0  # If conversion fails, return 0
    return 0  # Return 0 if value is missing

@app.route('/get_reports', methods=['GET', 'POST'])
def get_reports():
    if request.method == 'POST':
        blood_report = request.files.get('blood_report')

        if blood_report:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], blood_report.filename)
            blood_report.save(filepath)

            # Extract data from report
            extracted_df = process_medical_report(filepath)
            if extracted_df is None:
                return render_template('get_reports.html', result="Error extracting data from report.")

            try:
                # Load extracted CSV
                extracted_csv = f"extracted_reports/{os.path.splitext(blood_report.filename)[0]} extracted.csv"
                df = pd.read_csv(extracted_csv)

                patient_name_value = df[df['Parameter'] == 'Name']['Value'].values
                patient_name = patient_name_value[0] if patient_name_value.size > 0 else "Unknown"

                patient_gender_value = df[df['Parameter'] == 'Sex']['Value'].values
                patient_gender = "Male" if (patient_gender_value.size > 0 and str(patient_gender_value[0]).strip().lower() == 'male') else "Female"
                patient_age = extract_value(df, 'Age')
                
                # Store Report Date
                report_date = datetime.now().strftime("%B %d, %Y")  

                hemoglobin = extract_value(df, 'Hemoglobin (Hb)')
                glucose = extract_value(df, 'Glucose')
                blood_pressure = extract_value(df, 'Trestbps (Blood Pressure)')
                cholesterol = extract_value(df, 'Chol (Cholesterol)')


                # Extract Gender and Age
                sex_value = df[df['Parameter'] == 'Sex']['Value'].values
                gender = 0 if sex_value.size > 0 and str(sex_value[0]).strip().lower() == 'male' else 1  
                age = extract_value(df, 'Age')

                # --- 1. ANEMIA DETECTION ---
                hemoglobin = extract_value(df, 'Hemoglobin (Hb)')
                mch = extract_value(df, 'MCH (Mean Corpuscular Hemoglobin)')
                mchc = extract_value(df, 'MCHC (Mean Corpuscular Hemoglobin Concentration)')
                mcv = extract_value(df, 'MCV (Mean Corpuscular Volume)')

                anemia_data = np.array([[mcv, mchc, mch, hemoglobin, gender]])
                anemia_df = pd.DataFrame(anemia_data, columns=['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV'])
                anemia_df_scaled = scalers["Anemia"].transform(anemia_df)
                anemia_pred = models["Anemia"].predict(anemia_df_scaled)[0]

                if hemoglobin < 12.0:
                    anemia_final = "Detected (Low Hemoglobin Level - Possible Anemia)"
                elif anemia_pred == 1:
                    anemia_final = "Detected (Model Prediction - Possible Anemia)"
                else:
                    anemia_final = "Not Detected"

                # --- 2. DIABETES DETECTION ---
                glucose = extract_value(df, 'Glucose')
                blood_pressure = extract_value(df, 'Blood Pressure')
                diabetes_data = np.array([[extract_value(df, 'Pregnancies'), glucose,
                                           extract_value(df, 'Blood Pressure'),
                                           extract_value(df, 'Skin Thickness'),
                                           extract_value(df, 'Insulin'),
                                           extract_value(df, 'BMI'),
                                           extract_value(df, 'Diabetes Pedigree Function'),
                                           age]])
                
                diabetes_df = pd.DataFrame(diabetes_data, columns=['Pregnancies', 'Glucose', 'BloodPressure',
                                                                    'SkinThickness', 'Insulin', 'BMI',
                                                                      'DiabetesPedigreeFunction', 'Age'])
                diabetes_df_scaled = scalers["Diabetes"].transform(diabetes_df)
                diabetes_pred = models["Diabetes"].predict(diabetes_df_scaled)[0]

                if glucose >= 126:
                    diabetes_final = "Detected (High Glucose Level - Possible Diabetes)"
                elif diabetes_pred == 1:
                    diabetes_final = "Detected (Model Prediction - Possible Diabetes)"
                else:
                    diabetes_final = "Not Detected"

                # --- 3. THALASSEMIA DETECTION ---
                rdw = extract_value(df, 'RDW (Red Cell Distribution Width)')
                rbc_count = extract_value(df, 'RBC Count')

                thalassemia_data = np.array([[age, hemoglobin, mch, mchc, rdw, rbc_count]])
                thalassemia_df = pd.DataFrame(thalassemia_data, columns=['Age', 'Hb', 'MCH', 'MCHC', 'RDW', 'RBC count'])
                thalassemia_df_scaled = thalassemia_df.fillna(0).astype(float)  

                thalassemia_final = models["Thalassemia"].predict(thalassemia_df_scaled)[0]

                # Rule-based validation for Thalassemia
                if mch < 27.0 and mchc < 32.0:
                    thalassemia_final = "Detected (Low MCH & MCHC - Possible Thalassemia Minor)"
                else:
                    thalassemia_final = "Not Detected"

                # --- 4. HEART DISEASE DETECTION ---
                trestbps = extract_value(df, 'Trestbps (Blood Pressure)')
                chol = extract_value(df, 'Chol (Cholesterol)')
                thalach = extract_value(df, 'Thalach (Max Heart Rate)')
                oldpeak = extract_value(df, 'Oldpeak (Depression Induced by Exercise)')
                exang = extract_value(df, 'Exang (Exercise Induced Angina)')

                heart_data = np.array([[age, gender, trestbps, chol, extract_value(df, 'Fbs (Fasting Blood Sugar)'),
                                        extract_value(df, 'Restecg (Resting Electrocardiographic Results)'),
                                        thalach, exang, oldpeak, extract_value(df, 'CP_1 (Chest Pain Type 1)'),
                                        extract_value(df, 'CP_2 (Chest Pain Type 2)'), extract_value(df, 'CP_3 (Chest Pain Type 3)'),
                                        extract_value(df, 'Thal_2'), extract_value(df, 'Thal_3'),
                                        extract_value(df, 'Slope_1 (Slope of Peak Exercise ST Segment)'),
                                        extract_value(df, 'Slope_2')]])

                heart_data = np.append(heart_data, 0)  
                heart_data = np.append(heart_data, 0)  

                heart_columns = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'ca',
                                 'cp_1', 'cp_2', 'cp_3', 'thal_1', 'thal_2', 'thal_3', 'slope_1', 'slope_2']

                heart_df = pd.DataFrame(heart_data.reshape(1, -1), columns=heart_columns)
                heart_df = heart_df.fillna(0).astype(float)
                heart_df_scaled = scalers["Heart Disease"].transform(heart_df)

                heart_pred = models["Heart Disease"].predict(heart_df_scaled)[0]

                if trestbps > 140 or chol > 240:
                    heart_final = "Detected (High Blood Pressure/Cholesterol - Heart Disease Risk)"
                elif heart_pred == 1:
                    heart_final = "Detected (Model Prediction - Possible Heart Disease)"
                else:
                    heart_final = "Not Detected"
                if anemia_final == diabetes_final == thalassemia_final == heart_final == "Not Detected":
                 if patient_name and patient_gender and patient_age:  # Ensure values are not None
                    return render_template('results.html',
                               patient_name=patient_name,
                               patient_age=patient_age,
                               patient_gender=patient_gender,
                               report_date=report_date,
                               anemia=anemia_final,
                               diabetes=diabetes_final,
                               thalassemia=thalassemia_final,
                               heart_disease=heart_final,
                               hemoglobin=hemoglobin,
                               glucose=glucose,
                               blood_pressure=blood_pressure,
                               cholesterol=cholesterol,
                               show_donor_registration=True)  # Pass flag to show a "Become a Donor" button
                 else:
                   print("Error: Missing patient data for donor registration")


                return render_template('results.html',
                                       patient_name=patient_name,
                                       patient_age=patient_age,
                                       patient_gender=patient_gender,
                                       report_date=report_date,
                                       anemia=anemia_final,
                                       diabetes=diabetes_final,
                                       thalassemia=thalassemia_final,
                                       heart_disease=heart_final,
                                       hemoglobin=hemoglobin,
                                       glucose=glucose,
                                       blood_pressure=blood_pressure,
                                       cholesterol=cholesterol)

            except Exception as e:
                return render_template('get_reports.html', result=f"Error processing report: {str(e)}")

    return render_template('get_reports.html')

@app.route('/donor_registration', methods=['GET', 'POST'])
def donor_registration():
    # Retrieve values from GET request (Pre-filled by ML model)
    name = request.args.get('name', 'Unknown')
    gender = request.args.get('gender', 'Unknown')
    age = request.args.get('age', 'Unknown')

    if request.method == 'POST':
        # Get form values
        name = request.form.get('name', name)
        gender = request.form.get('gender', gender)
        age = request.form.get('age', age)
        email = request.form.get('email')
        phone = request.form.get('phone')
        city = request.form.get('city')
        blood_group = request.form.get('blood_group')  # NEW
        last_donation_date = request.form.get('last_donation_date')  # NEW

        # Debugging
        print(f"Received -> Name: {name}, Age: {age}, Gender: {gender}, Blood Group: {blood_group}, Last Donation: {last_donation_date}")

        if email and phone and city and blood_group:
            print(f"Saving donor: {name}, Blood Group: {blood_group}, Last Donation: {last_donation_date}")
            save_donor_to_db(name, age, gender, email, phone, city, blood_group, last_donation_date)
            return render_template('thank_you.html', message="Thank you! You have been registered as a donor.")

    return render_template('donor_registration.html', name=name, gender=gender, age=age)


def save_donor_to_db(name, age, gender, email, phone, city, blood_group, last_donation_date):
    # Handle missing values
    name = name if name else "Unknown"
    gender = gender if gender else "Unknown"
    age = age if age else "Unknown"
    last_donation_date = last_donation_date if last_donation_date else "Never Donated"

    print(f"Saving to DB -> Name: {name}, Age: {age}, Gender: {gender}, Blood Group: {blood_group}, Last Donation: {last_donation_date}")

    query = """
    CREATE (d:Donor {
        name: $name, 
        age: $age, 
        gender: $gender, 
        email: $email, 
        phone: $phone, 
        city: $city,
        blood_group: $blood_group,
        last_donation_date: $last_donation_date
    })
    RETURN d
    """

    result = db.query(query, {
        "name": name, "age": age, "gender": gender,
        "email": email, "phone": phone, "city": city,
        "blood_group": blood_group, "last_donation_date": last_donation_date
    })

    if result:
        print(f"Donor {name} saved successfully!")
    else:
        print("Error: Donor data was not saved!")



@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Donor Route
@app.route('/donor', methods=['GET', 'POST'])
def donor_check():
    if request.method == 'POST':
        try:
            # Check if basic details are submitted
            name = request.form.get('name')
            gender = request.form.get('gender')
            age = request.form.get('age')
            weight = request.form.get('weight')
            blood_group = request.form.get('blood_group')

            # Debugging the received form fields
            print(f"Received: Name={name}, Gender={gender}, Age={age}, Weight={weight}, Blood Group={blood_group}")

            if name and gender and age and weight and blood_group:
                # Processing basic details and saving them
                age = int(age)
                weight = float(weight)

                # Basic Eligibility Check for age and weight
                if age < 18 or age > 60:
                    return render_template('donor.html', result="You are not eligible to donate blood due to age restrictions.", show_health_form=False)
                if weight < 45:
                    return render_template('donor.html', result="You are not eligible to donate blood due to insufficient weight.", show_health_form=False)

                # If basic details are valid, show health-related questions
                return render_template('donor.html', show_health_form=True)

            else:
                return render_template('donor.html', result="Please fill all required fields.", show_health_form=False)

        except KeyError as e:
            return render_template('donor.html', result=f"Missing required field: {str(e)}", show_health_form=False)

        except ValueError as e:
            return render_template('donor.html', result="Please enter valid numeric values for age and weight.", show_health_form=False)

    return render_template('donor.html', show_health_form=False)

# Eligibility Route for Donor
@app.route('/eligibility', methods=['POST'])
def eligibility():
    if request.method == 'POST':
        try:
            # Health-related questions
            fever = request.form.get('fever')
            fever_type = request.form.get('fever_type')
            surgery = request.form.get('surgery')
            blood_thinners = request.form.get('blood_thinners')
            chronic_condition = request.form.get('chronic_condition')
            condition_management = request.form.get('condition_management')
            pregnancy = request.form.get('pregnancy')

            # Health Eligibility Checks
            if fever and fever.lower() == "yes" and fever_type == "severe":
                return render_template('donor.html', result="You are not eligible to donate blood due to severe illness in the last 7 days.", show_health_form=False)

            if surgery and surgery.lower() == "yes":
                return render_template('donor.html', result="You are not eligible to donate blood due to recent surgery.", show_health_form=False)

            if blood_thinners and blood_thinners.lower() == "yes":
                return render_template('donor.html', result="You are not eligible to donate blood due to medication or blood thinners.", show_health_form=False)

            if chronic_condition and chronic_condition.lower() == "yes" and condition_management.lower() == "no":
                return render_template('donor.html', result="You are not eligible to donate blood due to an unmanaged chronic condition.", show_health_form=False)

            if pregnancy and pregnancy.lower() == "yes":
                return render_template('donor.html', result="You are not eligible to donate blood due to pregnancy or breastfeeding.", show_health_form=False)

            # If eligible, redirect to blood report upload page
            return redirect(url_for('get_reports'))

        except Exception as e:
            return render_template('donor.html', result=f"An error occurred: {str(e)}", show_health_form=False)

    return render_template('donor.html', show_health_form=False)



@app.route('/register-patient', methods=['GET', 'POST'])
def register_patient():
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        gender = request.form.get('gender')
        dob = request.form.get('dob')
        blood_group = request.form.get('blood-group')
        mobile = request.form.get('mobile')
        email = request.form.get('email')
        city = request.form.get('city')
        address = request.form.get('address')
        reason = request.form.get('reason')
        emergency_contact = request.form.get('emergency-contact')

        # Save the patient data in Neo4j (No password field)
        query = """
        CREATE (p:Patient {
            name: $name, gender: $gender, dob: $dob, 
            blood_group: $blood_group, mobile: $mobile, 
            email: $email, city: $city, address: $address, 
            reason: $reason, emergency_contact: $emergency_contact
        })
        RETURN p
        """
        db.query(query, {
            "name": name, "gender": gender, "dob": dob,
            "blood_group": blood_group, "mobile": mobile,
            "email": email, "city": city, "address": address,
            "reason": reason, "emergency_contact": emergency_contact
        })

        # Prepare data for rendering (No password included)
        patient_data = {
            'name': name,
            'gender': gender,
            'dob': dob,
            'blood_group': blood_group,
            'mobile': mobile,
            'email': email,
            'city': city,
            'address': address,
            'reason': reason,
            'emergency_contact': emergency_contact
        }

        # Render the patient details page
        return render_template('patient_details.html', patient_data=patient_data)

    return render_template('patient_registration.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Blood compatibility dictionary
BLOOD_COMPATIBILITY = {
    "O-": ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"],
    "O+": ["O+", "A+", "B+", "AB+"],
    "A-": ["A-", "A+", "AB-", "AB+"],
    "A+": ["A+", "AB+"],
    "B-": ["B-", "B+", "AB-", "AB+"],
    "B+": ["B+", "AB+"],
    "AB-": ["AB-", "AB+"],
    "AB+": ["AB+"]
}

def get_compatible_blood_groups(blood_group):
    """Returns a list of blood groups that the given blood group can donate to."""
    return BLOOD_COMPATIBILITY.get(blood_group, [])


@app.route('/search_donor', methods=['GET'])
def search_donor():
    blood_group = request.args.get('blood-group')
    city = request.args.get('city')

    # Get compatible blood groups
    compatible_blood_groups = get_compatible_blood_groups(blood_group)

    # Query to find donors with compatible blood groups in Neo4j
    query = """
    MATCH (d:Donor)
    WHERE d.blood_group IN $compatible_blood_groups AND toLower(d.city) = toLower($city)
    RETURN d.name AS name, d.age AS age, d.gender AS gender, d.phone AS phone, d.email AS email, 
           d.blood_group AS blood_group, d.last_donation_date AS last_donation_date, d.city AS city
    """

    results = db.query(query, {"compatible_blood_groups": compatible_blood_groups, "city": city})

    donors = []
    for record in results:
        donors.append({
            "name": record["name"],
            "age": record["age"],
            "gender": record["gender"],
            "phone": record["phone"],
            "email": record["email"],
            "blood_group": record["blood_group"],
            "last_donation_date": record["last_donation_date"],
            "city": record["city"]
        })

    return render_template('final_results.html', donors=donors, blood_group=blood_group, city=city)



if __name__ == '__main__':
    app.run(debug=True)
