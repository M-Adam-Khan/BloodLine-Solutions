import pandas as pd
import pdfplumber
import pytesseract
from PIL import Image
import re
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text if text.strip() else None

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

def extract_text_from_file(file_path):
    file_extension = os.path.splitext(file_path)[-1].lower()
    
    if file_extension in ['.pdf']:
        return extract_text_from_pdf(file_path)
    elif file_extension in ['.jpg', '.jpeg', '.png']:
        return extract_text_from_image(file_path)
    elif file_extension in ['.txt']:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        print(f"Unsupported file format: {file_extension}")
        return None

def clean_and_structure_data(raw_text):
    if not raw_text:
        return None
    
    cleaned_text = re.sub(r'\(cid:\d+\)', '', raw_text)
    
    data_dict = {}
    for line in cleaned_text.strip().split("\n"):
        key_value = re.split(r":\s*", line, maxsplit=1)  
        if len(key_value) == 2:
            key, value = key_value
            value = re.sub(r'\(.*?\)', '', value).strip()
            data_dict[key.strip()] = float(value) if value.replace('.', '', 1).isdigit() else value

    return data_dict

def process_medical_report(file_path):
    raw_text = extract_text_from_file(file_path)
    structured_data = clean_and_structure_data(raw_text)

    if structured_data:
        df = pd.DataFrame(list(structured_data.items()), columns=["Parameter", "Value"])  
        
        output_folder = "extracted_reports"
        os.makedirs(output_folder, exist_ok=True)
        
        file_name = os.path.splitext(os.path.basename(file_path))[0]  
        output_file = os.path.join(output_folder, f"{file_name} extracted.csv")  

        df.to_csv(output_file, index=False)
        print(f"✅ Data saved as: {output_file}")
        
        return df
    else:
        print("No data extracted from the file.")
        return None

file_path = "uploads\Ali_cbc_report.pdf" 
df = process_medical_report(file_path)

if df is not None:
    print("\n✅ Extracted and Structured Data:")
    print(df.to_string(index=False)) 
