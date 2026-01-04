# 🤖 AI-Powered Email Enrichment Tool

This Streamlit app enriches email data by:
- Extracting **company names** and **sectors**
- Identifying **deadlines** with urgency status
- Detecting whether emails require **attention**

## 🚀 Setup Instructions

### 1. Install Python
Make sure you have **Python 3.9+** installed.

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
On Windows:
```bash
setup.bat
```

Or manually:
```bash
streamlit run app4.py
```

## ⚙️ Ollama Setup
This app uses [Ollama](https://ollama.ai/) to run local LLMs.  
- Install Ollama  
- Pull a model (example: `ollama pull llama3`)  
- Ensure Ollama is running before starting the app  

## 📂 Input Data
- Upload a `.csv` or `.xlsx` file containing email data.  
- Select which columns should be analyzed.  

## 📥 Output
- Interactive table with enriched information  
- Downloadable Excel file with results
