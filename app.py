import streamlit as st
import os
import time
import json
import unicodedata
import re
# from google.colab import userdata # Removed for Streamlit
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from groq import Groq
# NLTK imports can be removed if not used for pre-processing query for Text-to-SQL
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# --- Streamlit UI ---
st.set_page_config(page_title="Leoni_chat", layout="wide")

# --- Configuration ---
# For Streamlit, use st.secrets
# Ensure you have .streamlit/secrets.toml with your credentials
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_SERVICE_KEY = st.secrets["SUPABASE_SERVICE_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY]):
        raise ValueError("One or more secrets not found in st.secrets.")
    # print("Credentials loaded from Streamlit secrets.") # Goes to console
except Exception as e:
    st.error(f"Error loading secrets: {e}. Please ensure .streamlit/secrets.toml is configured.")
    st.stop()


# --- Model & DB Config ---
MARKDOWN_TABLE_NAME = "markdown_chunks"
ATTRIBUTE_TABLE_NAME = "Leoni_attributes"
RPC_FUNCTION_NAME = "match_markdown_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
GROQ_MODEL_FOR_SQL = "meta-llama/llama-4-maverick-17b-128e-instruct" # "mixtral-8x7b-32768" is also good
GROQ_MODEL_FOR_ANSWER = "meta-llama/llama-4-maverick-17b-128e-instruct" # "mixtral-8x7b-32768"

# --- Search Parameters ---
VECTOR_SIMILARITY_THRESHOLD = 0.60
VECTOR_MATCH_COUNT = 3

# --- Initialize Clients (Cached for performance) ---
@st.cache_resource
def init_supabase_client():
    try:
        client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("Supabase client initialized.") # Console log
        return client
    except Exception as e:
        st.error(f"Error initializing Supabase client: {e}")
        return None

@st.cache_resource
def load_sentence_transformer_model():
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"Sentence Transformer model ({EMBEDDING_MODEL_NAME}) loaded.") # Console log
        test_emb = model.encode("test")
        if len(test_emb) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"Embedding dimension mismatch. Expected {EMBEDDING_DIMENSIONS}, got {len(test_emb)}")
        return model
    except Exception as e:
        st.error(f"Error loading Sentence Transformer model: {e}")
        return None

@st.cache_resource
def init_groq_client():
    try:
        client = Groq(api_key=GROQ_API_KEY)
        print("Groq client initialized.") # Console log
        return client
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")
        return None

supabase = init_supabase_client()
st_model = load_sentence_transformer_model()
groq_client = init_groq_client()

if not all([supabase, st_model, groq_client]):
    st.error("One or more critical services could not be initialized. The chatbot cannot function.")
    st.stop()

# --- Helper Functions (Copied from your script, minor changes for logging) ---
def get_query_embedding(text):
    if not text or not st_model: return None
    try: return st_model.encode(text).tolist()
    except Exception as e: print(f"    Error generating query embedding: {e}"); return None

def find_relevant_markdown_chunks(query_embedding, supabase_client):
    if not query_embedding or not supabase_client: return []
    try:
        response = supabase_client.rpc(RPC_FUNCTION_NAME, {
            'query_embedding': query_embedding,
            'match_threshold': VECTOR_SIMILARITY_THRESHOLD,
            'match_count': VECTOR_MATCH_COUNT
        }).execute()
        return response.data if response.data else []
    except Exception as e: print(f"    Error calling RPC '{RPC_FUNCTION_NAME}': {e}"); return []

def generate_sql_from_query(user_query, table_schema, groq_cli, debug_prints):
    debug_prints.append("    Attempting Text-to-SQL generation...")
    leoni_attributes_schema = """
(id: bigint, Number: text, Name: text, "Object Type Indicator": text, Context: text, Version: text, State: text, "Last Modified": timestamp with time zone, "Created On": timestamp with time zone, "Sourcing Status": text, "Material Filling": text, "Material Name": text, "Max. Working Temperature [°C]": numeric, "Min. Working Temperature [°C]": numeric, Colour: text, "Contact Systems": text, Gender: text, "Housing Seal": text, "HV Qualified": text, "Length [mm]": numeric, "Mechanical Coding": text, "Number Of Cavities": numeric, "Number Of Rows": numeric, "Pre-assembled": text, Sealing: text, "Sealing Class": text, "Terminal Position Assurance": text, "Type Of Connector": text, "Width [mm]": numeric, "Wire Seal": text, "Connector Position Assurance": text, "Colour Coding": text, "Set/Kit": text, "Name Of Closed Cavities": text, "Pull-To-Seat": text, "Height [mm]": numeric, Classification: text)
"""
    prompt = f"""Your task is to convert natural language questions into PostgreSQL SELECT queries for the "Leoni_attributes" table.
Strictly adhere to the following rules:
1.  **Output Only SQL or NO_SQL:** Your entire response must be either a single, valid PostgreSQL SELECT statement ending with a semicolon (;) OR the exact word NO_SQL if the question cannot be answered by querying the table. Do not add explanations or markdown formatting.
2.  **Target Table:** ONLY query the "Leoni_attributes" table.
3.  **Column Quoting:** ONLY use double quotes around column names if they contain spaces, capital letters (besides the first letter if that's the only capital), or special characters like [, ], °, %. Standard names like "Number", "Name", "Context", "Version", "Colour", "Gender" usually DO NOT need quotes. When in doubt, check the schema provided. Example requiring quotes: "Object Type Indicator", "Max. Working Temperature [°C]".
4.  **SELECT Clause:**
    *   Select the columns explicitly asked for by the user.
    *   If the user's question implies a condition (e.g., "parts that are RED" or "parts with more than 10 cavities"), **you MUST include the column(s) involved in that condition in your SELECT statement**, in addition to any other columns the user explicitly requests. This allows verification of the condition. For example, for "parts that are red", include "Colour" in the SELECT. For "parts with more than 10 cavities", include "Number Of Cavities".
    *   If the user asks a general question about a part (e.g., "tell me about P00001636"), `SELECT *` is appropriate.
5.  **Filtering:**
    *   Use `=` for exact matches, especially for full part numbers provided in the "Number" column.
    *   Use `ILIKE` for case-insensitive text matching (e.g., `WHERE "Name" ILIKE '%housing%'`). Use wildcards `%` appropriately.
    *   Use `LIKE 'Value%'` or `ILIKE 'Value%'` for "starts with" queries on text fields.
    *   Use standard operators (`>`, `<`, `=`, `>=`, `<=`) for numeric and date/timestamp comparisons. Format dates as 'YYYY-MM-DD'.
    *   Combine conditions using `AND` / `OR` as needed.
6.  **LIMIT Clause:** Always include a `LIMIT` clause. Use `LIMIT 3` for queries targeting a specific, unique identifier like "Number". Use `LIMIT 10` for broader searches (like by name or color) unless the user explicitly asks for "all" matching items or a different quantity.
7.  **NO_SQL:** If the question asks for general knowledge, definitions found in separate documentation, or something clearly outside the scope of the table schema, respond ONLY with NO_SQL.

Table Schema: "Leoni_attributes"
{leoni_attributes_schema}

Examples:
User Question: "What is part number P00001636?"
SQL Query: SELECT * FROM "Leoni_attributes" WHERE "Number" = 'P00001636' LIMIT 3;

User Question: "Show me supplier parts containing 'connector'"
SQL Query: SELECT "Number", "Name", "Object Type Indicator" FROM "Leoni_attributes" WHERE "Object Type Indicator" = 'Supplier Part' AND "Name" ILIKE '%connector%' LIMIT 10;

User Question: "Find part numbers starting with C"
SQL Query: SELECT "Number", "Name" FROM "Leoni_attributes" WHERE "Number" ILIKE 'C%' LIMIT 10;

User Question: "What is the colour and version for 0-1718091-1?"
SQL Query: SELECT "Number", "Colour", "Version" FROM "Leoni_attributes" WHERE "Number" = '0-1718091-1' LIMIT 3;

User Question: "List part numbers that are black"
SQL Query: SELECT "Number", "Colour" FROM "Leoni_attributes" WHERE "Colour" ILIKE '%black%' LIMIT 10;

User Question: "List all distinct object type indicators"
SQL Query: SELECT DISTINCT "Object Type Indicator" FROM "Leoni_attributes";

User Question: "What is a TPA?"
SQL Query: NO_SQL

User Question: "{user_query}"
SQL Query:
"""
    if not groq_cli: return None
    try:
        response = groq_cli.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert Text-to-SQL assistant generating PostgreSQL queries."},
                {"role": "user", "content": prompt}
            ], model=GROQ_MODEL_FOR_SQL, temperature=0.0, max_tokens=500
        )
        if not response.choices or not response.choices[0].message: return None
        generated_sql = response.choices[0].message.content.strip()

        if generated_sql == "NO_SQL":
            debug_prints.append("    LLM determined no SQL query is applicable.")
            return None
        if generated_sql.upper().startswith("SELECT") and generated_sql.endswith(';'):
            forbidden = ["UPDATE", "DELETE", "INSERT", "DROP", "TRUNCATE", "ALTER", "CREATE", "EXECUTE", "GRANT", "REVOKE"]
            if any(k in generated_sql.upper() for k in forbidden):
                debug_prints.append(f"    WARNING: Generated SQL forbidden. Discarding: {generated_sql}")
                return None
            if ATTRIBUTE_TABLE_NAME.lower() not in generated_sql.lower():
                 debug_prints.append(f"    WARNING: Generated SQL not querying '{ATTRIBUTE_TABLE_NAME}'. Discarding: {generated_sql}")
                 return None
            debug_prints.append(f"    Generated SQL: {generated_sql}")
            return generated_sql
        else:
            debug_prints.append(f"    LLM response was not valid SQL or NO_SQL: {generated_sql}")
            return None
    except Exception as e:
        debug_prints.append(f"    Error during Text-to-SQL generation: {e}")
        return None

def find_relevant_attributes_with_sql(generated_sql, supabase_client, debug_prints):
    if not generated_sql or not supabase_client: return []
    try:
        debug_prints.append(f"    Attempting to execute generated SQL on '{ATTRIBUTE_TABLE_NAME}' (using placeholder logic)...")
        debug_prints.append(f"      SQL Generated by LLM: {generated_sql}")

        select_clause = "*"
        sel_match = re.match(r"SELECT\s+(.*?)\s+FROM", generated_sql, re.I | re.S)
        if sel_match:
            select_clause = sel_match.group(1).strip()
            if select_clause != "*" and '"Number"' not in select_clause and 'Number' not in select_clause :
                 if "distinct" not in select_clause.lower():
                    select_clause += ', "Number"'

        query = supabase_client.table(ATTRIBUTE_TABLE_NAME).select(select_clause)

        where_clause_full = ""
        where_match = re.search(r'WHERE\s+(.*?)(?:ORDER BY|LIMIT|$)', generated_sql, re.I | re.S)
        if where_match:
            where_clause_full = where_match.group(1).strip().rstrip(';')
            debug_prints.append(f"      Extracted WHERE (approx): {where_clause_full}")

        filter_pattern = re.compile(
            r'("?[\w\s\[\]%.°\-]+"?)\s*(ILIKE|LIKE|=)\s*\'(.*?)\'',
            re.IGNORECASE
        )
        conditions_applied_count = 0
        processed_conditions_for_logging = []

        for individual_condition_match in filter_pattern.finditer(where_clause_full):
            filter_col_name = individual_condition_match.group(1).replace('"', '')
            operator = individual_condition_match.group(2).upper()
            filter_value = individual_condition_match.group(3)
            processed_conditions_for_logging.append(f'`{filter_col_name}` {operator} \'{filter_value}\'')
            if operator == '=': query = query.eq(filter_col_name, filter_value)
            elif operator == 'ILIKE': query = query.ilike(filter_col_name, filter_value)
            elif operator == 'LIKE': query = query.like(filter_col_name, filter_value)
            else: continue
            conditions_applied_count +=1

        if conditions_applied_count > 0:
            debug_prints.append(f"      Applied {conditions_applied_count} filter condition(s) based on: " + ", ".join(processed_conditions_for_logging))
        elif where_clause_full:
            debug_prints.append("      WARNING: Could not translate complex WHERE clause with placeholder. Filters may not be fully applied.")
        else:
            debug_prints.append("      No WHERE clause detected or parsed.")

        final_limit_to_apply = 10
        limit_match = re.search(r'LIMIT\s*(\d+)', generated_sql, re.I | re.S)
        if limit_match:
            final_limit_to_apply = int(limit_match.group(1))
            debug_prints.append(f"      Limit parsed from SQL: {final_limit_to_apply}")
        else:
            debug_prints.append(f"      No LIMIT clause found in SQL by regex, using default: {final_limit_to_apply}")

        query = query.limit(final_limit_to_apply)
        debug_prints.append(f"      Executing Placeholder Query (approximated): SELECT {select_clause} FROM {ATTRIBUTE_TABLE_NAME} ... [LIMIT {final_limit_to_apply}]")
        response = query.execute()

        if response.data:
            debug_prints.append(f"      SQL query returned {len(response.data)} row(s).")
            # debug_prints.append("      Retrieved Data (first 3 rows):")
            # for i, row_data in enumerate(response.data[:3]):
            #      debug_prints.append(f"        Row {i+1}: {json.dumps(row_data, indent=2)}")
            # if len(response.data) > 3: debug_prints.append(f"        ... and {len(response.data)-3} more rows.")
            return response.data
        else:
            debug_prints.append("      SQL query returned no matching rows.")
            return []
    except Exception as e:
        debug_prints.append(f"    Error executing placeholder SQL query on '{ATTRIBUTE_TABLE_NAME}': {e}")
        return []

def format_context(markdown_chunks, attribute_rows):
    context_str = ""
    md_present = bool(markdown_chunks)
    attr_present = bool(attribute_rows)
    if md_present:
        context_str += "Context from LEOparts Standards Document:\n\n"
        for i, chunk in enumerate(markdown_chunks):
            filename = chunk.get('filename', 'Unknown Source')
            content = chunk.get('content', 'Content unavailable')
            similarity = chunk.get('similarity', None)
            context_str += f"--- Document Chunk {i+1} (Source: {filename}"
            if similarity is not None: context_str += f" | Similarity: {similarity:.4f}"
            context_str += ") ---\n" + content + "\n---\n\n"
    if attr_present:
        if md_present: context_str += "\n"
        context_str += "Context from Leoni Attributes Table:\n\n"
        for i, row in enumerate(attribute_rows):
            context_str += f"--- Attribute Row {i+1} ---\n"
            row_str_parts = []
            for key, value in row.items():
                if value is not None:
                    row_str_parts.append(f"  {key}: {value}")
            context_str += "\n".join(row_str_parts)
            context_str += "\n---\n\n"
    if not md_present and not attr_present:
        return "No relevant information found in the knowledge base (documents or attributes)."
    return context_str.strip()

def get_groq_chat_response(prompt, context_provided=True, groq_cli=None, debug_prints=None):
    if not groq_cli: return "Error: Groq client not available."
    if context_provided:
        system_message = "You are a helpful assistant knowledgeable about LEOparts standards and attributes. Answer the user's question based *only* on the provided context from the Standards Document and/or the Attributes Table. The Attributes Table context shows rows retrieved based on the user's query; assume these rows accurately reflect the query's conditions. Synthesize information from both sources if relevant and available. Be concise. If listing items, like part numbers, list them clearly."
    else:
        system_message = "You are a helpful assistant knowledgeable about LEOparts standards and attributes. You were unable to find relevant information in the knowledge base (documents or attributes) to answer the user's question. State clearly that the information is not available in the provided materials. Do not make up information or answer from general knowledge."
    try:
        response = groq_cli.chat.completions.create(
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
            model=GROQ_MODEL_FOR_ANSWER, temperature=0.1, stream=False )
        return response.choices[0].message.content
    except Exception as e:
        if debug_prints: debug_prints.append(f"    Error calling Groq API: {e}")
        return "Error contacting LLM."

# --- Main Chatbot Logic Function ---
leoni_attributes_schema_for_main_loop = """(id: bigint, Number: text, Name: text, "Object Type Indicator": text, Context: text, Version: text, State: text, "Last Modified": timestamp with time zone, "Created On": timestamp with time zone, "Sourcing Status": text, "Material Filling": text, "Material Name": text, "Max. Working Temperature [°C]": numeric, "Min. Working Temperature [°C]": numeric, Colour: text, "Contact Systems": text, Gender: text, "Housing Seal": text, "HV Qualified": text, "Length [mm]": numeric, "Mechanical Coding": text, "Number Of Cavities": numeric, "Number Of Rows": numeric, "Pre-assembled": text, Sealing: text, "Sealing Class": text, "Terminal Position Assurance": text, "Type Of Connector": text, "Width [mm]": numeric, "Wire Seal": text, "Connector Position Assurance": text, "Colour Coding": text, "Set/Kit": text, "Name Of Closed Cavities": text, "Pull-To-Seat": text, "Height [mm]": numeric, Classification: text)"""

def process_user_query(user_query):
    """
    Processes the user query and returns the bot's response and debug information.
    """
    debug_log = [] # To store intermediate processing messages

    relevant_markdown_chunks = []
    relevant_attribute_rows = []
    context_was_found = False
    generated_sql_for_debug = "N/A"

    # 1. Attempt Text-to-SQL generation
    generated_sql = generate_sql_from_query(user_query, leoni_attributes_schema_for_main_loop, groq_client, debug_log)
    if generated_sql:
        generated_sql_for_debug = generated_sql

    # 2. Execute SQL
    if generated_sql:
        relevant_attribute_rows = find_relevant_attributes_with_sql(generated_sql, supabase, debug_log)
        if relevant_attribute_rows: context_was_found = True
    else:
        debug_log.append(" -> Text-to-SQL generation failed or not applicable.")

    # 3. Perform Vector Search
    run_vector_search = True
    if run_vector_search:
        debug_log.append(" -> Generating query embedding for descriptive search...")
        query_embedding = get_query_embedding(user_query)
        if query_embedding:
            debug_log.append(f" -> Searching '{MARKDOWN_TABLE_NAME}' (Vector Search)...")
            relevant_markdown_chunks = find_relevant_markdown_chunks(query_embedding, supabase)
            if relevant_markdown_chunks: context_was_found = True
        else:
            debug_log.append("Error: Could not generate embedding. Skipping vector search.")

    # 4. Prepare Context
    context_str = format_context(relevant_markdown_chunks, relevant_attribute_rows)
    if not context_was_found:
        debug_log.append(" -> No relevant information found from either source.")
    else:
        debug_log.append(f" -> Found {len(relevant_markdown_chunks)} doc chunk(s) and {len(relevant_attribute_rows)} attribute row(s).")

    # 5. Generate Response
    debug_log.append(" -> Generating response with Groq...")
    prompt_for_llm = f"""Context:
{context_str}

User Question: {user_query}

Answer the user question based *only* on the provided context."""

    llm_response = get_groq_chat_response(prompt_for_llm, context_provided=context_was_found, groq_cli=groq_client, debug_prints=debug_log)

    # Prepare debug information for display
    debug_info_str = f"Generated SQL: {generated_sql_for_debug}\n"
    debug_info_str += f"Markdown Chunks Found: {len(relevant_markdown_chunks)}\n"
    debug_info_str += f"Attribute Rows Found: {len(relevant_attribute_rows)}\n\n"
    debug_info_str += "Processing Log:\n" + "\n".join(debug_log)

    return llm_response, debug_info_str

# --- Streamlit UI ---
st.title("Leoni_chat") # Header as requested

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug_info" not in st.session_state:
    st.session_state.debug_info = []


# Display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display debug info associated with this assistant message
        if message["role"] == "assistant" and i < len(st.session_state.debug_info) and st.session_state.debug_info[i]:
            with st.expander("Show Processing Details"):
                st.text(st.session_state.debug_info[i])


# Accept user input
if prompt := st.chat_input("Your Question:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Thinking..."):
            assistant_response, debug_details = process_user_query(prompt)
            full_response = assistant_response
        message_placeholder.markdown(full_response)
        with st.expander("Show Processing Details"): # Show details for the current response
            st.text(debug_details)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.debug_info.append(debug_details) # Store debug info

    # Store debug info for historical messages (needs to align with messages)
    # Pad debug_info if user messages don't have one
    if len(st.session_state.debug_info) < len(st.session_state.messages):
        # The last message was user, the one before that was assistant, and its debug info is already added
        # If we directly append user message, and then assistant message + its debug info,
        # the lengths should align: one debug_info per assistant message.
        # Let's adjust how debug_info is stored to be a list parallel to messages,
        # but only populated for assistant.
        # For simplicity, if the last message was assistant, its debug info is the last one added.

        # Re-aligning debug_info: create a list that has debug info for assistant messages, None otherwise
        aligned_debug_info = []
        debug_idx = 0
        for msg in st.session_state.messages:
            if msg["role"] == "assistant":
                if debug_idx < len(st.session_state.debug_info):
                    aligned_debug_info.append(st.session_state.debug_info[debug_idx])
                    debug_idx += 1
                else: # Should not happen if logic is correct, but as a fallback
                    aligned_debug_info.append(None)
            else:
                aligned_debug_info.append(None)
        st.session_state.debug_info = aligned_debug_info # Overwrite with aligned list

# A small footer or instruction
st.sidebar.markdown("---")
st.sidebar.markdown("Ask questions about LEOparts standards and attributes.")
st.sidebar.markdown(f"Using Groq SQL Model: `{GROQ_MODEL_FOR_SQL}`")
st.sidebar.markdown(f"Using Groq Answer Model: `{GROQ_MODEL_FOR_ANSWER}`")