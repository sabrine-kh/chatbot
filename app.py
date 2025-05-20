
# Cell 2: Main Script (Generalized Text-to-SQL Focused)
import ast
import os
import time
import json
import unicodedata
import re
from google.colab import userdata
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from groq import Groq
# NLTK imports can be removed if not used for pre-processing query for Text-to-SQL
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# --- Configuration ---
try:
    SUPABASE_URL         = userdata.get('SUPABASE_URL')
    SUPABASE_SERVICE_KEY = userdata.get('SUPABASE_SERVICE_KEY')
    GROQ_API_KEY         = userdata.get('GROQ_API_KEY')
    if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY]):
        raise ValueError("One or more secrets not found.")
    print("Credentials loaded from Colab secrets.")
except Exception as e:
    print(f"Error loading secrets: {e}")
    exit()

# --- Model & DB Config ---
MARKDOWN_TABLE_NAME    = "markdown_chunks"
ATTRIBUTE_TABLE_NAME   = "Leoni_attributes"          # <<< VERIFY
RPC_FUNCTION_NAME      = "match_markdown_chunks"     # <<< VERIFY
EMBEDDING_MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS   = 384

# ░░░  MODEL SWITCH  ░░░
GROQ_MODEL_FOR_SQL     = "qwen-qwq-32b"              ### <-- CHANGED
GROQ_MODEL_FOR_ANSWER  = "qwen-qwq-32b"              ### <-- CHANGED
print(f"Using Groq Model for SQL: {GROQ_MODEL_FOR_SQL}")
print(f"Using Groq Model for Answer: {GROQ_MODEL_FOR_ANSWER}")

# --- Search Parameters ---
VECTOR_SIMILARITY_THRESHOLD = 0.4
VECTOR_MATCH_COUNT          = 3

# --- Initialize Clients ---
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("Supabase client initialized.")
except Exception as e:
    print(f"Error initializing Supabase client: {e}")
    exit()

try:
    st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Sentence Transformer model ({EMBEDDING_MODEL_NAME}) loaded.")
    test_emb = st_model.encode("test")
    if len(test_emb) != EMBEDDING_DIMENSIONS:
        raise ValueError("Embedding dimension mismatch")
except Exception as e:
    print(f"Error loading Sentence Transformer model: {e}")
    exit()

try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("Groq client initialized.")
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    exit()

# ───────────────────────────────────────────────────────────────────────────
# HELPER TO STRIP <think> … </think> FROM GROQ RESPONSES
# ───────────────────────────────────────────────────────────────────────────
def strip_think_tags(text: str) -> str:
    """
    Removes any <think> … </think> block (case-insensitive, single or multiline)
    that the reasoning model may prepend to its answer.
    """
    if not text:
        return text
    return re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>',
                  '',
                  text,
                  flags=re.IGNORECASE | re.DOTALL).strip()

# ───────────────────────────────────────────────────────────────────────────
# Existing helper functions
# ───────────────────────────────────────────────────────────────────────────
import json
import ast

def _normalise_chunk(row):
    """
    Ensure each markdown chunk is a plain dict with at least a 'content' key.
    Handles four cases:
      1. Already the desired dict               → return as-is
      2. Wrapper   {'data': {...}}              → unwrap
      3. JSON/text '{"filename":...}'           → json.loads / ast.literal_eval
      4. Bare string 'Some paragraph …'         → wrap in {'content': ...}
    """
    # case 1
    if isinstance(row, dict) and "content" in row:
        return row

    # case 2 – single-key wrapper
    if isinstance(row, dict) and len(row) == 1:
        row = next(iter(row.values()))

    # case 3 – JSON string
    if isinstance(row, str):
        try:
            row = json.loads(row)
        except json.JSONDecodeError:
            try:
                row = ast.literal_eval(row)
            except Exception:
                # case 4 – treat as plain text
                return {"content": row, "filename": "Unknown", "similarity": None}

    # final guard
    if isinstance(row, dict):
        row.setdefault("filename", "Unknown")
        row.setdefault("similarity", None)
        return row

    # give up: return minimal stub so formatter won't crash
    return {"content": str(row), "filename": "Unknown", "similarity": None}

def get_query_embedding(text):
    if not text:
        return None
    try:
        return st_model.encode(text).tolist()
    except Exception as e:
        print(f"    Error generating query embedding: {e}")
        return None

def find_relevant_markdown_chunks(query_embedding):
    if not query_embedding:
        return []

    resp = supabase.rpc(
        RPC_FUNCTION_NAME,
        {
            'query_embedding': query_embedding,
            'match_threshold': VECTOR_SIMILARITY_THRESHOLD,
            'match_count': VECTOR_MATCH_COUNT
        }
    ).execute()

    raw_rows = resp.data or []
    print("      markdown RPC:", raw_rows[:3])   # optional debug

    return [_normalise_chunk(r) for r in raw_rows]


# ───────────────────────────────────────────────────────────────────────────
#  TEXT-TO-SQL GENERATION
# ───────────────────────────────────────────────────────────────────────────
def generate_sql_from_query(user_query, table_schema):
    """Uses Groq LLM with refined prompt and examples to generate SQL, attempting broad keyword matching."""
    print("    Attempting Text-to-SQL generation...")
    # --- Full Original Prompt ---
    prompt = f"""Your task is to convert natural language questions into robust PostgreSQL SELECT queries for the "Leoni_attributes" table. The primary goal is to find matching rows even if the user slightly misspells a keyword or uses variations.

Strictly adhere to the following rules:
1. **Output Only SQL or NO_SQL**: Your entire response must be either a single, valid PostgreSQL SELECT statement ending with a semicolon (;) OR the exact word NO_SQL if the question cannot be answered by querying the table. Do not add explanations or markdown formatting.
2. **Target Table**: ONLY query the "Leoni_attributes" table.
3. **Column Quoting**: Use double quotes around column names ONLY if necessary (contain spaces, capitals beyond first letter, special chars). Check schema: {table_schema}
4. **SELECT Clause**:
   - Select columns explicitly asked for or implied by the user's condition.
   - Always include the columns involved in the WHERE clause conditions for verification.
   - Use `SELECT *` for requests about a specific part number.
5. **Robust Keyword Searching (CRITICAL RULE)**:
   - Identify the main descriptive keyword(s) in the user's question (e.g., colors, materials, types like 'black', 'connector', 'grey', 'terminal'). Do NOT apply this robust search to specific identifiers like part numbers unless the user query implies a pattern search (e.g., 'starts with...').
   - For the identified keyword(s), generate a comprehensive list of **potential variations**:
     - **Common Abbreviations:** (e.g., 'blk', 'bk' for black; 'gry', 'gy' for grey; 'conn' for connector; 'term' for terminal).
     - **Alternative Spellings/Regional Variations:** (e.g., 'grey'/'gray', 'colour'/'color').
     - **Different Casings:** (e.g., 'BLK', 'Gry', 'CONN').
     - ***Likely Typos/Common Misspellings:*** (e.g., for 'black', consider 'blak', 'blck'; for 'terminal', consider 'termnial', 'terminl'; for 'connector', 'conecter'). Use your knowledge of common typing errors, but be reasonable – don't include highly improbable variations.
   - Search for the original keyword AND **ALL generated variations** across **multiple relevant text-based attributes**. Relevant attributes typically include "Colour", "Name", "Material Name", "Context", "Type Of Connector", "Terminal Position Assurance", etc. – use context to decide which columns are most relevant for the specific keyword.
   - Use `ILIKE` with surrounding wildcards (`%`) (e.g., `'%variation%'`) for case-insensitive, substring matching for every term and variation.
   - Combine **ALL** these individual search conditions (original + all variations across all relevant columns) using the `OR` operator. This might result in a long WHERE clause, which is expected.
6. **LIMIT Clause**: Use `LIMIT 3` for specific part number lookups. Use `LIMIT 10` (or maybe `LIMIT 20` if many variations are generated) for broader keyword searches to provide a reasonable sample.
7. **NO_SQL**: Return NO_SQL for general knowledge questions, requests outside the table's scope, or highly ambiguous queries.

Table Schema: "Leoni_attributes"
{table_schema}

Examples:
User Question: "What is part number P00001636?"
SQL Query: SELECT * FROM "Leoni_attributes" WHERE "Number" = 'P00001636' LIMIT 3;

User Question: "Show me supplier parts containing 'connector'"
SQL Query: SELECT "Number", "Name", "Object Type Indicator", "Type Of Connector" FROM "Leoni_attributes" WHERE "Object Type Indicator" = 'Supplier Part' AND ("Name" ILIKE '%connector%' OR "Name" ILIKE '%conn%' OR "Name" ILIKE '%conecter%' OR "Type Of Connector" ILIKE '%connector%' OR "Type Of Connector" ILIKE '%conn%' OR "Type Of Connector" ILIKE '%conecter%') LIMIT 10; # Includes variation and likely typo

User Question: "Find part numbers starting with C"
SQL Query: SELECT "Number", "Name" FROM "Leoni_attributes" WHERE "Number" ILIKE 'C%' LIMIT 10; # Pattern search, not robust keyword search

User Question: "List part numbers that are black"
SQL Query: SELECT "Number", "Colour", "Name", "Material Name" FROM "Leoni_attributes" WHERE "Colour" ILIKE '%black%' OR "Colour" ILIKE '%blk%' OR "Colour" ILIKE '%bk%' OR "Colour" ILIKE '%BLK%' OR "Colour" ILIKE '%blak%' OR "Colour" ILIKE '%blck%' OR "Name" ILIKE '%black%' OR "Name" ILIKE '%blk%' OR "Name" ILIKE '%bk%' OR "Name" ILIKE '%BLK%' OR "Name" ILIKE '%blak%' OR "Name" ILIKE '%blck%' OR "Material Name" ILIKE '%black%' OR "Material Name" ILIKE '%blk%' OR "Material Name" ILIKE '%bk%' OR "Material Name" ILIKE '%BLK%' OR "Material Name" ILIKE '%blak%' OR "Material Name" ILIKE '%blck%' LIMIT 10; # Example with typos added

User Question: "Any grey parts?"
SQL Query: SELECT "Number", "Colour", "Name" FROM "Leoni_attributes" WHERE "Colour" ILIKE '%grey%' OR "Colour" ILIKE '%gray%' OR "Colour" ILIKE '%gry%' OR "Colour" ILIKE '%gy%' OR "Colour" ILIKE '%GRY%' OR "Colour" ILIKE '%graey%' OR "Name" ILIKE '%grey%' OR "Name" ILIKE '%gray%' OR "Name" ILIKE '%gry%' OR "Name" ILIKE '%gy%' OR "Name" ILIKE '%GRY%' OR "Name" ILIKE '%graey%' LIMIT 10; # Example with alternative spelling, typo

User Question: "Parts with more than 10 cavities"
SQL Query: SELECT "Number", "Number Of Cavities" FROM "Leoni_attributes" WHERE "Number Of Cavities" > 10 LIMIT 10;

User Question: "What is a TPA?"
SQL Query: NO_SQL

User Question: "{user_query}"
SQL Query:
"""
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert Text-to-SQL assistant generating PostgreSQL queries optimized for finding matches despite keyword variations and typos."},
                {"role": "user", "content": prompt}
            ],
            model       = GROQ_MODEL_FOR_SQL,          ### <-- CHANGED
            temperature = 0.1,
            max_tokens  = 131072
        )
        if not response.choices or not response.choices[0].message:
            return None

        # ░░░ STRIP REASONING BLOCK FIRST ░░░
        generated_sql = strip_think_tags(response.choices[0].message.content) ### <-- CHANGED

        if generated_sql == "NO_SQL":
            print("    LLM determined no SQL query is applicable.")
            return None

        # Check if valid SQL (starts with SELECT, ends with ;)
        if generated_sql.upper().startswith("SELECT") and generated_sql.rstrip().endswith(';'): ### <-- CHANGED .rstrip()
            forbidden = ["UPDATE", "DELETE", "INSERT", "DROP", "TRUNCATE",
                         "ALTER", "CREATE", "EXECUTE", "GRANT", "REVOKE"]
            pattern = re.compile(r'\b(?:' + '|'.join(forbidden) + r')\b', re.IGNORECASE)
            if pattern.search(generated_sql):
             print(f"    WARNING: Generated SQL contains forbidden keyword. Discarding: {generated_sql}")
             return None


            # Check if the target table name appears after FROM
            table_name_pattern = r'FROM\s+(?:[\w]+\.)?("?' + ATTRIBUTE_TABLE_NAME + r'"?)'
            if not re.search(table_name_pattern, generated_sql, re.IGNORECASE):
                print(f"    WARNING: Generated SQL might not be querying '{ATTRIBUTE_TABLE_NAME}' correctly. Discarding: {generated_sql}")
                return None

            print(f"    Generated SQL: {generated_sql}")
            return generated_sql
        else:
            print(f"    LLM response was not valid SQL or NO_SQL: {generated_sql}")
            return None
    except Exception as e:
        print(f"    Error during Text-to-SQL generation: {e}")
        return None

# ───────────────────────────────────────────────────────────────────────────
#  SQL EXECUTION FUNCTION  (unchanged from original)
# ───────────────────────────────────────────────────────────────────────────


def _to_dict(maybe_json):
    """
    Ensure the value is a Python dict. Decode JSON/JSONB strings if needed.
    """
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        # Try fast JSON decode first
        try:
            return json.loads(maybe_json)
        except json.JSONDecodeError:
            pass
        # Fallback: literal_eval handles single quotes, etc.
        try:
            return ast.literal_eval(maybe_json)
        except Exception:
            pass
    # Give up – return an empty dict to avoid crashing format_context
    return {}

def find_relevant_attributes_with_sql(generated_sql: str):
    """
    Executes the LLM-generated SELECT via execute_readonly_sql().
    Always returns List[dict] rows.
    """
    if not generated_sql:
        return []

    sql_to_run = generated_sql.rstrip().rstrip(';')
    try:
        res = supabase.rpc("execute_readonly_sql", {"q": sql_to_run}).execute()

        # ─── raw dump for troubleshooting ───
        print("      ─── Raw rows from execute_readonly_sql ───")
        for i, raw in enumerate(res.data or []):
            print(f"      Row {i+1}: {raw}")
        print("      ───────────────────────────────────────────")

        if not res.data:
            return []

        # If each element is already a dict, just return the list as-is
        if isinstance(res.data[0], dict):
            return res.data

        # Otherwise grab the single JSON column
        first_key = next(iter(res.data[0].keys()))
        return [_to_dict(row[first_key]) for row in res.data]

    except Exception as e:
        print(f"    Error executing SQL via RPC: {e}")
        return []




# ───────────────────────────────────────────────────────────────────────────
#  CONTEXT FORMATTING (unchanged from original)
# ───────────────────────────────────────────────────────────────────────────
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
                    row_str_parts.append(f"  {key}: {json.dumps(value)}")
            context_str += "\n".join(row_str_parts)
            context_str += "\n---\n\n"
    if not md_present and not attr_present:
        return "No relevant information found in the knowledge base (documents or attributes)."
    return context_str.strip()

# ───────────────────────────────────────────────────────────────────────────
#  CHAT COMPLETION FOR ANSWERS
# ───────────────────────────────────────────────────────────────────────────
def get_groq_chat_response(prompt, context_provided=True):
    if context_provided:
        system_message = ("You are a helpful assistant knowledgeable about LEOparts standards and attributes. "
                          "Answer the user's question based *only* on the provided context from the Standards Document and/or the Attributes Table. "
                          "The Attributes Table context shows rows retrieved based on the user's query; assume these rows accurately reflect the query's conditions as interpreted by the client-side filters. "
                          "Synthesize information from both sources if relevant and available. Be concise. If listing items, like part numbers, list them clearly.")
    else:
        system_message = ("You are a helpful assistant knowledgeable about LEOparts standards and attributes. "
                          "You were unable to find relevant information in the knowledge base (documents or attributes) to answer the user's question. "
                          "State clearly that the information is not available in the provided materials. Do not make up information or answer from general knowledge.")

    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": prompt}
            ],
            model       = GROQ_MODEL_FOR_ANSWER,       ### <-- CHANGED
            temperature = 0.1,
            stream      = False
        )
        raw_reply = response.choices[0].message.content
        return strip_think_tags(raw_reply)            ### <-- CHANGED
    except Exception as e:
        print(f"    Error calling Groq API: {e}")
        return "Error contacting LLM."

# ───────────────────────────────────────────────────────────────────────────
#  MAIN CHAT LOOP
# ───────────────────────────────────────────────────────────────────────────
print("\n--- LEOparts Standards & Attributes Chatbot (Text-to-SQL Enabled) ---")

leoni_attributes_schema_for_main_loop = """(id: bigint, Number: text, Name: text, "Object Type Indicator": text, Context: text, Version: text, State: text, "Last Modified": timestamp with time zone, "Created On": timestamp with time zone, "Sourcing Status": text, "Material Filling": text, "Material Name": text, "Max. Working Temperature [°C]": numeric, "Min. Working Temperature [°C]": numeric, Colour: text, "Contact Systems": text, Gender: text, "Housing Seal": text, "HV Qualified": text, "Length [mm]": numeric, "Mechanical Coding": text, "Number Of Cavities": numeric, "Number Of Rows": numeric, "Pre-assembled": text, Sealing: text, "Sealing Class": text, "Terminal Position Assurance": text, "Type Of Connector": text, "Width [mm]": numeric, "Wire Seal": text, "Connector Position Assurance": text, "Colour Coding": text, "Set/Kit": text, "Name Of Closed Cavities": text, "Pull-To-Seat": text, "Height [mm]": numeric, Classification: text)"""

while True:
    user_query = input("\nYour Question: ")
    if user_query.lower() == 'quit':
        break
    if not user_query.strip():
        continue

    relevant_markdown_chunks = []
    relevant_attribute_rows  = []
    context_was_found        = False
    generated_sql            = None

    # 1. Attempt Text-to-SQL generation
    generated_sql = generate_sql_from_query(user_query, leoni_attributes_schema_for_main_loop)

    # 2. Execute SQL (using client-side filters)
    if generated_sql:
        relevant_attribute_rows = find_relevant_attributes_with_sql(generated_sql)
        if relevant_attribute_rows:
            context_was_found = True
    else:
        print(" -> Text-to-SQL generation failed or not applicable.")

    # 3. Perform Vector Search (can be conditional)
    run_vector_search = True
    # Example: Potentially skip if SQL found *exactly* one specific item and the query was precise
    # if generated_sql and len(relevant_attribute_rows) == 1 and "what is part number" in user_query.lower():
       # run_vector_search = False

    if run_vector_search:
        print(" -> Generating query embedding for descriptive search...")
        query_embedding = get_query_embedding(user_query)
        if query_embedding:
            print(f" -> Searching '{MARKDOWN_TABLE_NAME}' (Vector Search)...")
            relevant_markdown_chunks = find_relevant_markdown_chunks(query_embedding)
            if relevant_markdown_chunks:
                context_was_found = True
        else:
            print("Error: Could not generate embedding. Skipping vector search.")

    # 4. Prepare Context
    context_str = format_context(relevant_markdown_chunks, relevant_attribute_rows)
    if not context_was_found:
        print(" -> No relevant information found from either source.")
    else:
        print(f" -> Found {len(relevant_markdown_chunks)} doc chunk(s) and {len(relevant_attribute_rows)} attribute row(s).")

    # 5. Generate Response
    print(" -> Generating response with Groq...")
    prompt_for_llm = f"""Context:
{context_str}

User Question: {user_query}

Answer the user question based *only* on the provided context."""
    llm_response = get_groq_chat_response(prompt_for_llm, context_provided=context_was_found)
    print("\nAssistant Response:")
    print(llm_response)