# Building a Gemini Data Analysis Agent - Step by Step Guide

This guide will walk you through creating an AI-powered data analysis tool using Google's Gemini model, Streamlit, and DuckDB.

## Step 1: Setting Up Google AI Studio and Getting API Key

1. Visit Google AI Studio
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account
   - If you don't have access, join the waitlist at [Google AI Studio Waitlist](https://makersuite.google.com/waitlist)

2. Create an API Key
   - Click on "Get API Key" in the top navigation
   - Select "Create API Key in new project" or use an existing project
   - Copy the generated API key and store it securely
   - Note: The API key starts with "AIza..."

## Step 2: Setting Up Your Development Environment

1. Create a new project directory:
```bash
mkdir data-analysis-agent
cd data-analysis-agent
```

2. Set up a Python virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Create the project structure:
```bash
touch app.py
touch requirements.txt
touch .env
touch .gitignore
```

4. Add the following to `.gitignore`:
```
venv/
.env
__pycache__/
*.pyc
.DS_Store
```

## Step 3: Installing Dependencies

1. Add the following to `requirements.txt`:
```
streamlit>=1.24.0
pandas>=1.5.3
google-cloud-aiplatform>=1.36.0
duckdb>=0.9.2
python-dotenv>=1.0.0
google-generativeai>=0.3.2
plotly>=5.18.0
openpyxl>=3.1.2
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Step 4: Setting Up Environment Variables

1. Create a `.env` file in your project root:
```bash
GOOGLE_API_KEY=your_api_key_here
```

## Step 5: Building the Application

Let's build the application step by step in `app.py`:

1. First, import the required libraries:
```python
import streamlit as st
import pandas as pd
import google.generativeai as genai
import duckdb
import tempfile
import csv
import json
import plotly.express as px
from typing import Tuple, List, Optional
import os
from dotenv import load_dotenv
```

2. Set up environment variables and configure Gemini:
```python
# Load environment variables
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
```

3. Implement file preprocessing function:
```python
def preprocess_and_save(file) -> Tuple[Optional[str], Optional[List[str]], Optional[pd.DataFrame]]:
    """Preprocess the uploaded file and save it as CSV."""
    try:
        # Handle different file types
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        
        # Clean and format data
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
        # Handle dates and numbers
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None
```

4. Implement SQL query generation:
```python
def generate_sql_query(question: str, schema: dict) -> str:
    """Generate SQL query using Gemini."""
    model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-1219')
    
    prompt = f"""You are an expert data analyst. Given the following question and database schema, 
    generate a SQL query that answers the question. The query should be valid DuckDB SQL.
    
    Schema: {json.dumps(schema, indent=2)}
    Question: {question}
    
    Return only the SQL query, nothing else."""
    
    response = model.generate_content(prompt)
    return response.text.strip()
```

5. Implement query execution:
```python
def execute_query(query: str, conn) -> pd.DataFrame:
    """Execute SQL query and return results."""
    try:
        return conn.execute(query).df()
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return pd.DataFrame()
```

6. Implement result analysis:
```python
def analyze_results(df: pd.DataFrame, question: str) -> str:
    """Analyze query results using Gemini."""
    model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-1219')
    
    prompt = f"""Analyze the following data results and provide insights in response to the question.
    Keep the analysis concise and focus on the most important findings.
    
    Question: {question}
    Data Summary: {df.describe().to_string()}
    
    First few rows: {df.head().to_string()}"""
    
    response = model.generate_content(prompt)
    return response.text
```

7. Implement the main application:
```python
def main():
    st.title("📊 Data Analyst Agent (Gemini)")
    st.write("Upload your data and ask questions in plain English!")
    
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        temp_path, columns, df = preprocess_and_save(uploaded_file)
        
        if temp_path and columns and df is not None:
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            # Create schema for the uploaded data
            schema = {
                "tables": [{
                    "name": "uploaded_data",
                    "columns": [{"name": col, "type": str(df[col].dtype)} for col in df.columns],
                    "description": "Contains the uploaded dataset"
                }]
            }
            
            # Initialize DuckDB connection
            conn = duckdb.connect(database=':memory:')
            conn.execute(f"CREATE TABLE uploaded_data AS SELECT * FROM read_csv_auto('{temp_path}')")
            
            # Query interface
            question = st.text_area("Ask a question about your data:", 
                                  placeholder="e.g., What is the average value? Show me the top 5 records...")
            
            if st.button("Analyze"):
                if question:
                    with st.spinner("Generating SQL query..."):
                        sql_query = generate_sql_query(question, schema)
                        st.code(sql_query, language="sql")
                    
                    with st.spinner("Executing query..."):
                        results_df = execute_query(sql_query, conn)
                        if not results_df.empty:
                            st.write("### Results")
                            st.dataframe(results_df)
                            
                            # Generate analysis
                            with st.spinner("Analyzing results..."):
                                analysis = analyze_results(results_df, question)
                                st.write("### Analysis")
                                st.write(analysis)
                            
                            # Attempt to create a visualization
                            if len(results_df) > 0 and len(results_df.columns) >= 2:
                                try:
                                    if results_df.select_dtypes(include=['number']).columns.any():
                                        fig = px.line(results_df) if len(results_df) > 1 else px.bar(results_df)
                                        st.plotly_chart(fig)
                                except Exception as e:
                                    st.warning("Couldn't generate visualization automatically.")
                else:
                    st.warning("Please enter a question!")

if __name__ == "__main__":
    main()
```

## Step 6: Running the Application

1. Start the Streamlit server:
```bash
streamlit run app.py
```

2. Access the application:
   - Open browser
   - Navigate to http://localhost:8501
   - Upload your data file
   - Start asking questions!

## Step 7: Example Usage

1. Sample questions to try:
   - "What is the average value of [column]?"
   - "Show me the top 5 records sorted by [column]"
   - "How many unique values are in [column]?"
   - "What is the distribution of [column]?"
   - "Show me the trend of [column] over time"

2. Example data format (sample.csv):
```csv
date,product,sales,quantity
2024-01-01,Product A,100.50,5
2024-01-02,Product B,75.25,3
2024-01-03,Product A,150.75,7
```

## Step 8: Troubleshooting Common Issues

1. API Key Issues:
   - Ensure `.env` file is in the correct location
   - Check API key format
   - Verify API key permissions

2. File Upload Issues:
   - Check file format
   - Verify file encoding
   - Check for special characters

3. Query Issues:
   - Check column names
   - Verify data types
   - Look for syntax errors
   - Check for unsupported operations

4. Visualization Issues:
   - Check data types
   - Verify number of records
   - Check for missing values

## Resources

1. Documentation:
   - [Streamlit Documentation](https://docs.streamlit.io/)
   - [Google AI Studio Documentation](https://ai.google.dev/)
   - [DuckDB Documentation](https://duckdb.org/docs/)
   - [Plotly Documentation](https://plotly.com/python/)

2. Community:
   - [Streamlit Forums](https://discuss.streamlit.io/)
   - [Google AI Discord](https://discord.gg/google-ai)
   - [DuckDB Slack](https://duckdb.org/community)

3. Additional Learning:
   - [SQL Tutorial](https://www.w3schools.com/sql/)
   - [Python Data Analysis](https://pandas.pydata.org/docs/getting_started/) 