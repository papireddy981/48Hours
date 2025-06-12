from google import genai 
from dotenv import load_dotenv
import os 
from google.genai import types 
import json 
import pickle 
import numpy as np
import pandas as pd 
import streamlit as st 
import re 
from fpdf import FPDF
import streamlit as st
import tempfile

st.title("Payroll Fraud Alerts using AI")

payroll_file = st.file_uploader("Upload Payroll Data")

def get_genai_response(payroll_data):

    load_dotenv()

    GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
    GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")

    client = genai.Client(
            vertexai=True,
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION,
            http_options=types.HttpOptions(api_version='v1')
    )

    system_instructions = """
        You are a financial audit assistant that explains why a given payroll record appears to be fraudulent.
        
        You will be provided with a JSON object containing payroll details for a single employee. Use financial and HR knowledge to analyze the record and provide a clear and concise explanation in natural language.
        
        Possible fraud indicators include (but are not limited to):
        - High overtime hours
        - Unusually large bonuses relative to salary
        - Salary paid after employee has resigned or been terminated
        - Shared bank accounts (used by multiple employees)
        
        Always explain in simple, human-readable terms as if you are writing a short note to a manager.
        
        If the record is not suspicious (column name "is_suspicious" if 0 that means not suspicious, else suspicious), say: "No fraud detected in this record."
        
        Return only the fraud type, explanation, and Resolution Suggestion in this format 
        Fraud Type :""
        Explanation :""
        Resolution Suggestion:""

        if no fraud is found, return None for fraud type, No fraud in explanation. None for Resolution.
        Do not repeat the input or label it.
    """


    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=payroll_data,
            config=types.GenerateContentConfig(
                system_instruction=system_instructions,
                max_output_tokens=256,
                temperature=0.3,
            )
        )

        return response.text 

    except Exception as e:
        print(f"Error in generating the explanation: {e}") 

def parse_genai_response(response):
    explanation = fraud_type = resolution_suggestion = None

    fraud_type_match = re.search(r'Fraud Type\s*:\s*"(.*?)"', response, re.DOTALL | re.IGNORECASE)
    explanation_match = re.search(r'Explanation\s*:\s*"(.*?)"', response, re.DOTALL | re.IGNORECASE)
    resolution_match = re.search(r'Resolution Suggestion\s*:\s*"(.*?)"', response, re.DOTALL | re.IGNORECASE) 

    if fraud_type_match:
        fraud_type = fraud_type_match.group(1).strip()
    if explanation_match:
        explanation = explanation_match.group(1).strip()
    if resolution_match:
        resolution_suggestion = resolution_match.group(1).strip()

    return fraud_type, explanation, resolution_suggestion



import math
from fpdf import FPDF

def get_multicell_height(pdf, text, width, line_height=10):
    """
    Helper to calculate height required for a multi_cell block given text and cell width.
    Uses multi_cell in split_only mode.
    """
    current_x = pdf.get_x()
    current_y = pdf.get_y()
    # split_only returns the list of lines without printing them.
    lines = pdf.multi_cell(width, line_height, text, border=0, align='L', split_only=True)
    pdf.set_xy(current_x, current_y)
    # If no lines detected, return line_height as default.
    return line_height * len(lines) if lines else line_height

def dataframe_to_pdf(df, pdf_path):
    """
    Export a pandas DataFrame to a PDF file with dynamic row heights and customized column widths.
    - Uses A3 landscape format for increased overall width.
    - Custom column weightings are used:
         * Small columns (e.g., "employee id", "working hours", "overtime hours", "is_suspicious") get lower weight.
         * Wide columns ("fraud type", "reason", "resolution suggestion") get higher weight.
         * All other columns use default weighting.
    Args:
        df: pandas DataFrame
        pdf_path: path to save the output PDF.
    """
    # Use A3 landscape (A3: 420 x 297 mm)
    pdf = FPDF(orientation='L', unit='mm', format='A3')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=8)

    # Define custom weightings for columns based on their header (using lower-case keys)
    small_cols = {"employee id", "working hours", "overtime hours", "is_suspicious"}
    wide_cols = {"fraud type", "reason", "resolution suggestion", "fraud explanation"}
    # Default weight for remaining columns.
    weights = []
    for col in df.columns:
        col_l = col.strip().lower()
        if col_l in small_cols:
            weights.append(0.8)
        elif col_l in wide_cols:
            weights.append(2.5)
        else:
            weights.append(1.0)
    
    total_weight = sum(weights)
    # Get usable page width.
    page_width = pdf.w - 2 * pdf.l_margin
    # Calculate width for each column based on its weight.
    col_widths = [ (w / total_weight) * page_width for w in weights ]

    # --- Add table header ---
    row_height = 10
    for idx, col in enumerate(df.columns):
        pdf.cell(col_widths[idx], row_height, str(col), border=1, align='C')
    pdf.ln(row_height)

    # --- Add table rows with dynamic row heights ---
    # We use custom wide columns list same as above.
    for _, row in df.iterrows():
        # Determine required cell height for each cell.
        cell_heights = []
        for idx, item in enumerate(row):
            col_name = df.columns[idx].strip().lower()
            if col_name in wide_cols:
                # Calculate height using the helper.
                cell_height = get_multicell_height(pdf, str(item), col_widths[idx], line_height=10)
            else:
                cell_height = row_height
            cell_heights.append(cell_height)
        max_height = max(cell_heights)

        # Save starting x, y for the row.
        y_start = pdf.get_y()
        for idx, item in enumerate(row):
            x_before = pdf.get_x()
            y_before = pdf.get_y()
            col_name = df.columns[idx].strip().lower()
            # For wide columns, use multi_cell. Otherwise, use cell.
            if col_name in wide_cols:
                pdf.multi_cell(col_widths[idx], 10, str(item), border=1, align='L')
                # Reset x to next cell if multi_cell wraps text.
                pdf.set_xy(x_before + col_widths[idx], y_before)
            else:
                pdf.cell(col_widths[idx], max_height, str(item), border=1)
        pdf.ln(max_height)
    pdf.output(pdf_path)

# Example usage:
# dataframe_to_pdf(your_dataframe, "output.pdf")
def data_clean(data):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    try:
        data['overtime_ratio'] = data['Overtime Hours'] / data['Working Hours']
    except Exception as e:
        data['overtime_ratio'] = 1e5

    data['Total Compensation'] = data['Salary'] + data['Bonuses']
    data['Is_Active'] = data['Employment Status'].apply(lambda x: 1 if x=='Employee' else 0)
    data['Bank Account Number'] = data['Bank Account Number'].str.extract('(\d+)', expand=False).astype(int)

    # Drop columns not needed for model input (if needed for prediction)
    X = data.drop(columns=["Employee ID", "Employee Name", "Employment Status", "Suspicious Activity Flag"], errors='ignore', inplace=False)
    x_scaled = scaler.transform(X)
    predictions = model.predict(x_scaled)
    data['is_suspicious'] = predictions

    # For each row, obtain GenAI response and extract fields
    for i in range(data.shape[0]):
        json_str = json.dumps(data.iloc[i].to_dict())
        response = get_genai_response(json_str)
        fraud_type, reason, resolution = parse_genai_response(response)
        data.loc[i, 'reason'] = reason
        data.loc[i, 'fraud type'] = fraud_type
        data.loc[i, 'resolution suggestion'] = resolution

    # Drop extra columns not needed in the PDF output
    df = data.drop(columns="Suspicious Activity Flag", errors='ignore')
    # Select only columns you want to show in the PDF. Adjust column names as needed.
    df_pdf = df.drop(columns=["Bank Account Number", "Is_Active", "Total Compensation", "overtime_ratio"], errors='ignore')

    st.write(df)

    # Generate PDF using the updated dataframe_to_pdf with custom widths
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:  # Assuming pdf_export.py is in /src
        dataframe_to_pdf(df_pdf, tmpfile.name)
        tmpfile.seek(0)
        st.download_button(
            label="Download results as PDF",
            data=tmpfile.read(),
            file_name="payroll_fraud_results.pdf",
            mime="application/pdf"
        )
    print(df)


if payroll_file is not None:
    dataframe = pd.read_csv(payroll_file)
    data_clean(dataframe.iloc[:20, :])


