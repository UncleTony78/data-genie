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

# Load environment variables
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

def preprocess_and_save(file) -> Tuple[Optional[str], Optional[List[str]], Optional[pd.DataFrame]]:
    """Preprocess the uploaded file and save it as CSV."""
    try:
        # Handle different file types
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        
        # Clean column names: remove extra spaces and standardize
        df.columns = [col.strip().replace('"', '') for col in df.columns]  # Remove quotes and spaces
        
        # Create a mapping of old to new column names to handle spaces consistently
        column_mapping = {col: col.strip() for col in df.columns}
        df = df.rename(columns=column_mapping)
        
        # Clean and format data
        for col in df.select_dtypes(include=['object']):
            # Replace 'None', 'NaN', 'null' with actual None
            df[col] = df[col].replace(['None', 'NaN', 'null', 'NULL', ''], None)
            # Clean string values (only for non-null values)
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Handle dates and numbers
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            # Clean numeric columns
            elif any(word in col.lower() for word in ['price', 'sales', 'profit', 'cost', 'discount', 'units']):
                try:
                    # First replace None-like values with NaN
                    df[col] = df[col].replace(['None', 'NaN', 'null', 'NULL', ''], pd.NA)
                    # Convert to numeric, coercing errors to NaN
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace('$', '').str.replace(',', ''),
                        errors='coerce'
                    )
                except Exception as e:
                    st.warning(f"Warning: Could not convert column '{col}' to numeric. Error: {e}")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False)
        
        return temp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

def generate_sql_query(question: str, schema: dict) -> str:
    """Generate SQL query using Gemini."""
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    prompt = f"""You are an expert SQL analyst. Your task is to generate a valid DuckDB SQL query.

    Database Schema:
    {json.dumps(schema, indent=2)}

    User Question: {question}

    Requirements:
    1. Generate ONLY the SQL query, no explanations
    2. Ensure the query is valid DuckDB SQL syntax
    3. Use the table name 'uploaded_data'
    4. Keep the query simple and focused
    5. Handle potential NULL values appropriately
    6. Column names have been cleaned (no extra spaces or quotes)
    7. Use the exact column names as shown in the schema
    8. For the overview query, include:
       - Total rows
       - Unique values in key columns
       - Min/Max/Avg of numeric columns
       - Date range if dates exist

    Example query for dataset overview:
    SELECT 
        COUNT(*) as total_rows,
        COUNT(DISTINCT Segment) as unique_segments,
        COUNT(DISTINCT Country) as unique_countries,
        MIN(Date) as earliest_date,
        MAX(Date) as latest_date,
        AVG("Units Sold") as avg_units_sold,
        SUM("Gross Sales") as total_sales
    FROM uploaded_data;

    SQL Query:"""
    
    try:
        response = model.generate_content(prompt, generation_config={
            'temperature': 0.1,
            'top_p': 0.8,
            'top_k': 40
        })
        # Extract and clean the SQL query
        sql_query = ''.join(part.text for part in response.parts).strip()
        # Remove any markdown code block syntax if present
        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        return sql_query
    except Exception as e:
        st.error(f"Error generating SQL query: {e}")
        return ""

def execute_query(query: str, conn) -> pd.DataFrame:
    """Execute SQL query and return results."""
    try:
        if not query:
            return pd.DataFrame()
        return conn.execute(query).df()
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return pd.DataFrame()

def analyze_results(df: pd.DataFrame, question: str) -> str:
    """Analyze query results using Gemini."""
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    prompt = f"""As a data analyst, analyze these results and provide clear insights.

    Original Question: {question}

    Data Statistics:
    {df.describe().to_string()}

    Sample Data (First Few Rows):
    {df.head().to_string()}

    Please provide:
    1. Direct answer to the question
    2. Key insights from the data
    3. Any notable patterns or trends
    4. Keep the analysis concise and clear

    Analysis:"""
    
    try:
        response = model.generate_content(prompt, generation_config={
            'temperature': 0.2,
            'top_p': 0.8,
            'top_k': 40
        })
        return ''.join(part.text for part in response.parts)
    except Exception as e:
        st.error(f"Error analyzing results: {e}")
        return "Unable to generate analysis."

def get_column_summary(df: pd.DataFrame) -> dict:
    """Generate a summary of each column in the dataframe."""
    summary = {}
    for column in df.columns:
        col_type = str(df[column].dtype)
        unique_count = df[column].nunique()
        null_count = df[column].isnull().sum()
        sample_values = df[column].dropna().head(3).tolist()
        
        summary[column] = {
            "type": col_type,
            "unique_values": unique_count,
            "null_count": null_count,
            "sample_values": sample_values
        }
    return summary

def main():
    st.title("📊 Data Analyst Agent (Gemini)")
    st.write("Upload your data and ask questions in plain English!")
    
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        temp_path, columns, df = preprocess_and_save(uploaded_file)
        
        if temp_path and columns and df is not None:
            # Data Preview Section
            st.write("### Data Overview")
            st.write(f"- Total Rows: {len(df)}")
            st.write(f"- Total Columns: {len(df.columns)}")
            
            # Display column names
            st.write("### Uploaded columns:")
            st.json(columns)  # This will display the columns in a format similar to your screenshot
            
            # Data Preview
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            # Get column summary before creating schema
            column_summary = get_column_summary(df)
            
            # Create enhanced schema for the uploaded data
            schema = {
                "tables": [{
                    "name": "uploaded_data",
                    "columns": [
                        {
                            "name": col,
                            "type": str(df[col].dtype),
                            "description": f"Contains {column_summary[col]['unique_values']} unique values, {column_summary[col]['null_count']} null values. Sample values: {', '.join(map(str, column_summary[col]['sample_values']))}"
                        } for col in df.columns
                    ],
                    "total_rows": len(df),
                    "description": f"Contains the uploaded dataset with {len(df)} rows and {len(df.columns)} columns."
                }]
            }
            
            # Initialize DuckDB connection
            conn = duckdb.connect(database=':memory:')
            conn.execute(f"CREATE TABLE uploaded_data AS SELECT * FROM read_csv_auto('{temp_path}')")
            
            # Query interface with context
            st.write("### Ask Questions")
            st.write("You can ask questions about:")
            st.write("- Column statistics (average, sum, count, etc.)")
            st.write("- Data filtering and grouping")
            st.write("- Trends and patterns")
            st.write("- Relationships between columns")
            
            question = st.text_area(
                "Ask a question about your data:", 
                placeholder="e.g., What is the average value? Show me the top 5 records..."
            )
            
            if st.button("Analyze"):
                if question:
                    with st.spinner("Generating SQL query..."):
                        sql_query = generate_sql_query(question, schema)
                        st.write("### Generated SQL Query")
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
                                        st.write("### Visualization")
                                        fig = px.line(results_df) if len(results_df) > 1 else px.bar(results_df)
                                        st.plotly_chart(fig)
                                except Exception as e:
                                    st.warning("Couldn't generate visualization automatically.")
                else:
                    st.warning("Please enter a question!")

if __name__ == "__main__":
    main() 