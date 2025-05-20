import streamlit as st
import os
from dotenv import load_dotenv
import sys
from io import StringIO
import contextlib

# Load environment variables from .env file for local development
load_dotenv()

# --- Configuration ---
# Try to get credentials from Streamlit secrets first, then fall back to environment variables
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv('SUPABASE_URL'))
SUPABASE_SERVICE_KEY = st.secrets.get("SUPABASE_SERVICE_KEY", os.getenv('SUPABASE_SERVICE_KEY'))
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv('GROQ_API_KEY'))

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY]):
    st.error("""
    Missing required credentials. Please ensure they are set in either:
    1. Streamlit Cloud Secrets (recommended for deployment)
    2. .env file (for local development)
    
    Required credentials:
    - SUPABASE_URL
    - SUPABASE_SERVICE_KEY
    - GROQ_API_KEY
    """)
    st.stop()

# Set environment variables for the main script
os.environ['SUPABASE_URL'] = SUPABASE_URL
os.environ['SUPABASE_SERVICE_KEY'] = SUPABASE_SERVICE_KEY
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

# Initialize Streamlit
st.set_page_config(
    page_title="LEOparts Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS to handle loading states
st.markdown("""
    <style>
    .stSpinner > div {
        text-align: center;
        align-items: center;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("LEOparts Standards & Attributes Chatbot")
st.markdown("Ask questions about LEOparts standards and attributes.")

# Initialize chat history and debug output
if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug_output" not in st.session_state:
    st.session_state.debug_output = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the query
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Capture stdout to get the response and debug output
            captured_output = StringIO()
            with contextlib.redirect_stdout(captured_output):
                try:
                    # Import and run the main script
                    from app import (
                        generate_sql_from_query,
                        find_relevant_attributes_with_sql,
                        get_query_embedding,
                        find_relevant_markdown_chunks,
                        format_context,
                        get_groq_chat_response,
                        leoni_attributes_schema_for_main_loop
                    )

                    # Process the query using the imported functions
                    relevant_markdown_chunks = []
                    relevant_attribute_rows = []
                    context_was_found = False
                    generated_sql = None

                    # 1. Attempt Text-to-SQL generation
                    generated_sql = generate_sql_from_query(prompt, leoni_attributes_schema_for_main_loop)

                    # 2. Execute SQL
                    if generated_sql:
                        relevant_attribute_rows = find_relevant_attributes_with_sql(generated_sql)
                        if relevant_attribute_rows:
                            context_was_found = True

                    # 3. Perform Vector Search
                    query_embedding = get_query_embedding(prompt)
                    if query_embedding:
                        relevant_markdown_chunks = find_relevant_markdown_chunks(query_embedding)
                        if relevant_markdown_chunks:
                            context_was_found = True

                    # 4. Prepare Context
                    context_str = format_context(relevant_markdown_chunks, relevant_attribute_rows)

                    # 5. Generate Response
                    prompt_for_llm = f"""Context:
{context_str}

User Question: {prompt}

Answer the user question based *only* on the provided context."""
                    
                    response = get_groq_chat_response(prompt_for_llm, context_provided=context_was_found)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    response = "I apologize, but I encountered an error while processing your request."

            # Get debug output
            debug_output = captured_output.getvalue()
            st.session_state.debug_output.append(debug_output)

            # Display the response
            st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Show debug information in an expandable section
            with st.expander("Debug Information"):
                st.code(debug_output)

# Add a sidebar with information about the models being used
with st.sidebar:
    st.header("Model Information")
    st.markdown("""
    - **SQL Generation Model**: qwen-qwq-32b
    - **Answer Generation Model**: qwen-qwq-32b
    - **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
    """)
    
    st.header("Search Parameters")
    st.markdown("""
    - **Vector Similarity Threshold**: 0.4
    - **Vector Match Count**: 3
    """) 