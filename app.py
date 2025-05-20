import streamlit as st
import ast
import os
import time
import json
import unicodedata
import re
from supabase import create_client
from sentence_transformers import SentenceTransformer
import groq

# Initialize Streamlit
st.set_page_config(page_title="LEOparts Chatbot", page_icon="⚙️")

# --- Configuration ---
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

# Initialize clients
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
groq_client = groq.Groq(api_key=GROQ_API_KEY)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Constants
RPC_FUNCTION_NAME = "match_markdown_chunks"
VECTOR_SIMILARITY_THRESHOLD = 0.4
VECTOR_MATCH_COUNT = 3

# Schema for the database
leoni_attributes_schema_for_main_loop = """
CREATE TABLE leoni_attributes (
    id SERIAL PRIMARY KEY,
    attribute_name VARCHAR(255),
    attribute_value TEXT,
    description TEXT,
    category VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def _normalise_chunk(chunk):
    """Normalize text by removing special characters and extra whitespace."""
    text = unicodedata.normalize('NFKD', chunk)
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_query_embedding(query):
    """Generate embedding for the query using the sentence transformer model."""
    try:
        query_embedding = embedding_model.encode(query)
        return query_embedding.tolist()
    except Exception as e:
        st.error(f"Error generating query embedding: {str(e)}")
        return None

def find_relevant_markdown_chunks(query_embedding):
    """Find relevant markdown chunks using vector similarity search."""
    try:
        response = supabase.rpc(
            RPC_FUNCTION_NAME,
            {
                "query_embedding": query_embedding,
                "match_threshold": VECTOR_SIMILARITY_THRESHOLD,
                "match_count": VECTOR_MATCH_COUNT
            }
        ).execute()
        
        if response.data:
            return [chunk['content'] for chunk in response.data]
        return []
    except Exception as e:
        st.error(f"Error finding relevant chunks: {str(e)}")
        return []

def generate_sql_from_query(query, schema):
    """Generate SQL query from natural language using Groq."""
    try:
        prompt = f"""Given this database schema:
{schema}

Generate a SQL query for this question: {query}

Return ONLY the SQL query, nothing else."""
        
        response = groq_client.chat.completions.create(
            model="qwen-qwq-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100
        )
        
        sql_query = response.choices[0].message.content.strip()
        return sql_query
    except Exception as e:
        st.error(f"Error generating SQL: {str(e)}")
        return None

def _to_dict(row):
    """Convert database row to dictionary."""
    return {
        'attribute_name': row['attribute_name'],
        'attribute_value': row['attribute_value'],
        'description': row['description'],
        'category': row['category']
    }

def find_relevant_attributes_with_sql(sql_query):
    """Execute SQL query and return relevant attributes."""
    try:
        response = supabase.table('leoni_attributes').select('*').execute()
        if response.data:
            return [_to_dict(row) for row in response.data]
        return []
    except Exception as e:
        st.error(f"Error executing SQL query: {str(e)}")
        return []

def format_context(markdown_chunks, attribute_rows):
    """Format the context from markdown chunks and attribute rows."""
    context_parts = []
    
    if markdown_chunks:
        context_parts.append("Relevant Documentation:")
        for chunk in markdown_chunks:
            context_parts.append(f"- {chunk}")
    
    if attribute_rows:
        context_parts.append("\nRelevant Attributes:")
        for row in attribute_rows:
            context_parts.append(f"- {row['attribute_name']}: {row['attribute_value']}")
            if row['description']:
                context_parts.append(f"  Description: {row['description']}")
    
    return "\n".join(context_parts)

def get_groq_chat_response(prompt, context_provided=True):
    """Get response from Groq chat model."""
    try:
        response = groq_client.chat.completions.create(
            model="qwen-qwq-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error getting chat response: {str(e)}")
        return "I apologize, but I encountered an error while processing your request."

def answer_question(question):
    """Main function to process a question and return an answer."""
    # 1. Get query embedding
    query_embedding = get_query_embedding(question)
    if not query_embedding:
        return "I apologize, but I couldn't process your question at this time."

    # 2. Find relevant markdown chunks
    relevant_markdown_chunks = find_relevant_markdown_chunks(query_embedding)
    
    # 3. Generate and execute SQL query
    generated_sql = generate_sql_from_query(question, leoni_attributes_schema_for_main_loop)
    relevant_attribute_rows = []
    if generated_sql:
        relevant_attribute_rows = find_relevant_attributes_with_sql(generated_sql)
    
    # 4. Format context
    context_str = format_context(relevant_markdown_chunks, relevant_attribute_rows)
    
    # 5. Generate response
    prompt_for_llm = f"""Context:
{context_str}

User Question: {question}

Answer the user question based *only* on the provided context."""
    
    return get_groq_chat_response(prompt_for_llm, context_provided=bool(context_str))

# Streamlit UI
st.title("⚙️  LEOparts Standards & Attributes Chatbot")

with st.form(key="qa_form"):
    user_q = st.text_input(
        "Ask a question about parts, colours, cavities…",
        placeholder="e.g. Any white parts?"
    )
    submitted = st.form_submit_button("Ask")

if submitted and user_q.strip():
    with st.spinner("Thinking…"):
        reply = answer_question(user_q)
    st.markdown("#### Answer")
    st.write(reply)
