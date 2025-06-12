from google import genai 
from dotenv import load_dotenv
import os 
from google.genai import types 
import json 
import pickle 
import numpy as np
import pandas as pd 
import streamlit as st 


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
        
        Return only the explanation. Do not repeat the input or label it.

        Manual inspection reasons : {}
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

    X = data.drop(columns=["Employee ID", "Employee Name", "Employment Status","Suspicious Activity Flag"], inplace=False) 

    print(X)
    x_scaled = scaler.transform(X) 

    predictions = model.predict(x_scaled)
    print(predictions)

    data['is_suspicious'] = predictions

    # st.write(data)
    # print(data)

    # print(data.iloc[0])

    for i in range(data.shape[0]):
        json_str = json.dumps(data.iloc[i].to_dict())
        response = get_genai_response(json_str)

        data.loc[i, 'reason'] = response

    df = data.drop(columns="Suspicious Activity Flag", inplace=False)

    st.write(df)
    print(df)



if payroll_file is not None:
    dataframe = pd.read_csv(payroll_file)
    # st.write(dataframe)

    data_clean(dataframe.iloc[:10, :])
