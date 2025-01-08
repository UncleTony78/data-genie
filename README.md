# Data Genie 🧞‍♂️

An AI-powered data analysis tool that allows you to analyze CSV and Excel files using natural language queries. Built with Google's Gemini model, this tool translates plain English questions into SQL queries, making data analysis accessible to everyone – no SQL expertise required.

## Features

- 📤 File Upload Support
  - CSV and Excel files
  - Automatic data type detection
  - Schema inference

- 💬 Natural Language Queries
  - Convert English questions to SQL
  - Get instant answers about your data
  - No SQL knowledge required

- 🔍 Advanced Analysis
  - Complex data aggregations
  - Filtering and sorting
  - Statistical summaries
  - Automatic visualizations

## Setup

1. Clone the repository:
```bash
git clone https://github.com/UncleTony78/data-genie.git
cd data-genie
```

2. Create a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Get your API Key:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key

5. Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_api_key_here
```

6. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Upload your CSV or Excel file
2. View the column names and data preview
3. Ask questions in plain English
4. Get instant analysis and visualizations

## Example Questions

- "What is the average value of [column]?"
- "Show me the top 5 records sorted by [column]"
- "How many unique values are in [column]?"
- "What is the distribution of [column]?"
- "Show me the trend of [column] over time"

## Requirements

- Python 3.10 or higher
- Internet connection (for Gemini API access)
- Google API key
- Supported file formats: CSV, Excel (.xlsx)

## Contributing

Feel free to open issues or submit pull requests with improvements.

## License

MIT License - feel free to use this project for your own purposes. 