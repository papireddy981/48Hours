# Payroll Fraud Alerts Using GenAI

## Overview

The Payroll Fraud Detection System is a comprehensive solution designed to identify and explain potential fraudulent activities in payroll data. This system integrates a machine learning model for fraud detection with Google's GenAI for generating human-readable explanations of suspicious records. The solution is divided into two main components: GenAI Integration and Model Training.

## Features

- **AI-Powered Fraud Detection**: Utilizes Google's GenAI to analyze payroll data and identify potential fraud indicators.
- **AI-Powered Explanations**: Utilizes GenAI to analyze payroll data and generate explanations for detected fraud.
- **Customizable Analysis**: Allows users to upload payroll data and receive detailed explanations of any detected fraud with resolution suggestions.
- **PDF Report Generation**: Generates a comprehensive PDF report of the analysis, which can be downloaded for further review.

## Installation

1. **Clone the Repository**: Clone the project repository to your local machine.
   ```bash
   git clone https://github.com/papireddy981/48Hours.git
   ```
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   ```
   Activate it
   ```bash
   venv\scripts\activate
   ```
2. **Install Requirements**
  ```bash
  pip install -r requirements.txt
```
3. **Train the model**
   Run all the cells in model.ipynb. A model.pkl will be downloaded along with scaler.pkl (StandardScaler)

4. **Run streamlit**
   ```bash
   streamlit run genai.py
   ```
5. **Upload a payroll file (csv)**

# Conclusion
The Payroll Fraud Detection System is a powerful tool for financial auditors, providing AI-driven insights into payroll data. By integrating a machine learning model with GenAI, the system not only detects potential fraud but also explains it in a clear and concise manner, aiding in effective decision-making and fraud prevention. This documentation provides a comprehensive overview of both the GenAI Integration and Model Training components, ensuring users can effectively utilize the system for fraud detection and analysis.
