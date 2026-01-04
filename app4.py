import streamlit as st
import pandas as pd
import ollama
import json
from io import BytesIO
from datetime import datetime, timedelta
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Data Enrichment Tool ✨",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def parse_and_calculate_deadline_status(deadline_text):
    """
    Parse deadline text and calculate the status based on days remaining.
    Returns: (deadline_date, status, days_remaining)
    """
    if not deadline_text or deadline_text.lower() in ['none', 'null', 'no deadline']:
        return None, "No Deadline", None
    
    try:
        # Try to parse various date formats
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{1,2}/\d{1,2}/\d{4}',  # M/D/YYYY
        ]
        
        date_str = None
        for pattern in date_patterns:
            match = re.search(pattern, deadline_text)
            if match:
                date_str = match.group()
                break
        
        if not date_str:
            return deadline_text, "Invalid Date", None
        
        # Parse the date
        try:
            if '-' in date_str and len(date_str.split('-')[0]) == 4:
                deadline_date = datetime.strptime(date_str, '%Y-%m-%d')
            elif '/' in date_str:
                if len(date_str.split('/')[2]) == 4:
                    deadline_date = datetime.strptime(date_str, '%m/%d/%Y')
                else:
                    deadline_date = datetime.strptime(date_str, '%m/%d/%y')
            elif '-' in date_str:
                deadline_date = datetime.strptime(date_str, '%m-%d-%Y')
            else:
                return deadline_text, "Invalid Format", None
        except ValueError:
            return deadline_text, "Invalid Date", None
        
        # Calculate days remaining
        today = datetime.now()
        days_remaining = (deadline_date - today).days
        
        # Determine status
        if days_remaining < 0:
            status = "Overdue"
        elif days_remaining < 3:
            status = "Urgent"  # Red
        elif days_remaining < 7:
            status = "Soon"    # Yellow
        else:
            status = "Future"  # Green
            
        return deadline_date.strftime('%Y-%m-%d'), status, days_remaining
        
    except Exception as e:
        return deadline_text, "Parse Error", None

def get_status_color(status):
    """Return color based on deadline status."""
    color_map = {
        "Urgent": "🔴",
        "Soon": "🟡", 
        "Future": "🟢",
        "Overdue": "⚫",
        "No Deadline": "⚪",
        "Invalid Date": "❓",
        "Invalid Format": "❓",
        "Parse Error": "❓"
    }
    return color_map.get(status, "❓")

# --- Enhanced LLM Function ---
def get_details_from_llm(text: str, model: str):
    """
    Uses Ollama to extract company name, sector, deadline, and attention requirement.
    """
    if not text or text.strip() == "|":
        return {
            "company_name": None, 
            "sector": None, 
            "deadline": "No deadline",
            "attention_required": "No"
        }

    system_prompt = """
    You are a highly skilled email analysis assistant. Your task is to analyze the provided email data and extract:
    1. Main company name (ignore personal names)
    2. Business sector
    3. Any deadline or due date mentioned
    4. Whether this email requires immediate attention/reply

    Rules:
    1. Focus only on company names, ignore personal names
    2. Provide the most likely sector (Technology, Finance, Retail, Healthcare, Education, etc.)
    3. Extract any dates that represent deadlines, due dates, or time-sensitive requirements
    4. Determine if email needs attention based on:
       - Urgent language ("urgent", "asap", "deadline", "immediate")
       - Questions that need answers
       - Action items or tasks
       - Meeting requests without responses
       - Complaints or issues
    5. If no clear value is present, return appropriate null/default values
    6. Respond ONLY with valid JSON

    Examples:

    Input: "LinkedIn <notifications@linkedin.com> | Meeting deadline: 2024-12-15 | Please confirm your attendance ASAP"
    Output: {"company_name": "LinkedIn", "sector": "Technology", "deadline": "2024-12-15", "attention_required": "Yes"}

    Input: "NVIDIA <news@nvidia.com> | Newsletter: Latest GPU updates"
    Output: {"company_name": "NVIDIA", "sector": "Technology", "deadline": "No deadline", "attention_required": "No"}
    
    Input: "Meeting with akshagr10@gmail.com | Project due next Friday | Need your feedback urgently"
    Output: {"company_name": null, "sector": null, "deadline": "next Friday", "attention_required": "Yes"}

    Input: "Bank of America <alerts@bankofamerica.com> | Payment due 01/15/2025 | Action required"
    Output: {"company_name": "Bank of America", "sector": "Finance", "deadline": "01/15/2025", "attention_required": "Yes"}
    """
    
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f'Input: """{text}"""'}
            ],
            format='json'
        )
        result = json.loads(response['message']['content'])
        
        # Ensure all required fields are present
        required_fields = ["company_name", "sector", "deadline", "attention_required"]
        for field in required_fields:
            if field not in result:
                result[field] = None if field in ["company_name", "sector"] else ("No deadline" if field == "deadline" else "No")
                
        return result
    except Exception as e:
        st.error(f"An error occurred with the LLM: {e}")
        return {
            "company_name": "LLM Error", 
            "sector": str(e),
            "deadline": "No deadline",
            "attention_required": "Error"
        }

# --- Helper Function for Excel Download ---
def to_excel(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Enriched_Data')
    return output.getvalue()

# --- Session State Initialization ---
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'errors' not in st.session_state:
    st.session_state.errors = []

# --- UI Layout ---
with st.sidebar:
    st.title("About")
    st.info(
        """
        This tool enriches email data by analyzing content using AI to identify:
        - Company name and sector
        - Deadlines with color-coded urgency
        - Attention requirements for emails
        """
    )
    st.title("⚙️ Configuration")
    try:
        models_list = [m['name'] for m in ollama.list()['models']]
        if not models_list:
            st.warning("No models found in Ollama.")
            models_list = ["llama3:8b"]
    except Exception:
        models_list = ["llama3:8b", "mistral", "gemma:7b"]
        st.warning("Could not connect to Ollama. Using default models list.")

    selected_model = st.selectbox(
        "Select the Ollama model to use:",
        options=models_list,
        help="Choose a model that is fast and good at JSON output, like Llama3."
    )
    
    st.markdown("---")
    st.markdown("**Legend:**")
    st.markdown("🔴 Urgent (< 3 days)")
    st.markdown("🟡 Soon (< 7 days)")
    st.markdown("🟢 Future (> 7 days)")
    st.markdown("⚫ Overdue")
    st.markdown("⚪ No deadline")

st.title("🤖 AI-Powered Email Enrichment Tool")
st.markdown("Upload a CSV or Excel file with email data. The AI will analyze each row to find company info, deadlines, and attention requirements.")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["csv", "xlsx"],
    help="Your file can have multiple columns. The AI will use them as context for analysis."
)

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("1. Data Preview & Column Selection")
        st.dataframe(df.head())

        context_columns = st.multiselect(
            "Which columns should the AI analyze for context?",
            options=df.columns,
            default=list(df.columns)
        )

        if context_columns:
            st.subheader("2. Start Processing")
            if st.button("✨ Enrich Data", type="primary"):
                st.session_state.processing_complete = False
                st.session_state.errors = []
                progress_bar = st.progress(0, text="Starting process...")
                
                # Initialize new columns
                df['company_name'] = ''
                df['sector'] = ''
                df['deadline'] = ''
                df['deadline_status'] = ''
                df['deadline_icon'] = ''
                df['attention_required'] = ''
                
                total_rows = len(df)

                for index, row in df.iterrows():
                    context_text = " | ".join(str(row[col]) for col in context_columns if pd.notna(row[col]))
                    status_text = f"Processing row {index + 1}/{total_rows}"
                    progress_bar.progress((index + 1) / total_rows, text=status_text)
                    
                    try:
                        details = get_details_from_llm(context_text, selected_model)
                        
                        # Basic info
                        df.at[index, 'company_name'] = details.get('company_name')
                        df.at[index, 'sector'] = details.get('sector')
                        df.at[index, 'attention_required'] = details.get('attention_required', 'No')
                        
                        # Process deadline
                        deadline_text = details.get('deadline', 'No deadline')
                        df.at[index, 'deadline'] = deadline_text
                        
                        # Parse and analyze deadline
                        parsed_date, status, days_remaining = parse_and_calculate_deadline_status(deadline_text)
                        df.at[index, 'deadline_status'] = status
                        df.at[index, 'deadline_icon'] = get_status_color(status)
                        
                        if days_remaining is not None:
                            if days_remaining >= 0:
                                df.at[index, 'deadline_status'] = f"{status} ({days_remaining} days)"
                            else:
                                df.at[index, 'deadline_status'] = f"Overdue ({abs(days_remaining)} days ago)"
                        
                    except Exception as e:
                        error_msg = f"Row {index+1}: Failed to process. Reason: {e}"
                        st.session_state.errors.append(error_msg)
                        df.at[index, 'company_name'] = 'Error'
                        df.at[index, 'sector'] = str(e)
                        df.at[index, 'deadline'] = 'Error'
                        df.at[index, 'deadline_status'] = 'Error'
                        df.at[index, 'deadline_icon'] = '❓'
                        df.at[index, 'attention_required'] = 'Error'
                
                progress_bar.empty()
                st.session_state.processed_df = df
                st.session_state.processing_complete = True

    except Exception as e:
        st.error(f"Failed to read the file. Please ensure it's a valid CSV or Excel file. Error: {e}")

if st.session_state.processing_complete:
    st.subheader("3. Results")
    st.success("Enrichment complete!")
    
    # Display results with colored formatting
    df_display = st.session_state.processed_df.copy()
    
    # Create a formatted display
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True
    )
    
    # --- Enhanced Results Summary ---
    st.subheader("📊 Results Summary")
    df_results = st.session_state.processed_df
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Emails", len(df_results))
        if 'attention_required' in df_results.columns:
            attention_count = len(df_results[df_results['attention_required'].str.lower() == 'yes'])
            st.metric("Require Attention", attention_count)
    
    with col2:
        if 'deadline_status' in df_results.columns:
            urgent_count = len(df_results[df_results['deadline_status'].str.contains('Urgent', na=False)])
            st.metric("Urgent Deadlines", urgent_count)
            overdue_count = len(df_results[df_results['deadline_status'].str.contains('Overdue', na=False)])
            st.metric("Overdue Items", overdue_count)
    
    with col3:
        if 'company_name' in df_results.columns:
            company_count = len(df_results[df_results['company_name'].notna() & (df_results['company_name'] != '') & (df_results['company_name'] != 'Error')])
            st.metric("Companies Identified", company_count)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'sector' in df_results.columns and not df_results['sector'].dropna().empty:
            sector_counts = df_results['sector'].value_counts().dropna()
            st.write("**Emails by Sector:**")
            st.bar_chart(sector_counts)
    
    with col2:
        if 'deadline_status' in df_results.columns:
            # Extract just the status part (before parentheses)
            status_clean = df_results['deadline_status'].str.extract(r'^([^(]+)')[0].str.strip()
            status_counts = status_clean.value_counts().dropna()
            st.write("**Deadline Status Distribution:**")
            st.bar_chart(status_counts)
    
    # Attention required breakdown
    if 'attention_required' in df_results.columns:
        st.subheader("🚨 Attention Required Analysis")
        attention_df = df_results[df_results['attention_required'].str.lower() == 'yes']
        if not attention_df.empty:
            st.write(f"Found {len(attention_df)} emails requiring attention:")
            st.dataframe(
                attention_df[['company_name', 'sector', 'deadline', 'deadline_icon', 'deadline_status', 'attention_required']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No emails currently require immediate attention.")
    
    # Download enhanced data
    excel_data = to_excel(st.session_state.processed_df)
    st.download_button(
        label="📥 Download Enhanced Excel File",
        data=excel_data,
        file_name=f"enhanced_email_data_{uploaded_file.name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    if st.session_state.errors:
        with st.expander("⚠️ View Processing Errors"):
            for error in st.session_state.errors:
                st.warning(error)