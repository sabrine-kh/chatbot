
# Cell 2: Main Script (Generalized Text-to-SQL Focused)
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
    SUPABASE_URL = userdata.get('SUPABASE_URL')
    SUPABASE_SERVICE_KEY = userdata.get('SUPABASE_SERVICE_KEY')
    GROQ_API_KEY = userdata.get('GROQ_API_KEY')
    if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY]):
        raise ValueError("One or more secrets not found.")
    print("Credentials loaded from Colab secrets.")
except Exception as e: print(f"Error loading secrets: {e}"); exit()

# --- Model & DB Config ---
MARKDOWN_TABLE_NAME = "markdown_chunks"
ATTRIBUTE_TABLE_NAME = "Leoni_attributes" # <<< VERIFY
RPC_FUNCTION_NAME = "match_markdown_chunks" # <<< VERIFY
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
GROQ_MODEL_FOR_SQL = "meta-llama/llama-4-maverick-17b-128e-instruct"
GROQ_MODEL_FOR_ANSWER = "meta-llama/llama-4-maverick-17b-128e-instruct"
print(f"Using Groq Model for SQL: {GROQ_MODEL_FOR_SQL}")
print(f"Using Groq Model for Answer: {GROQ_MODEL_FOR_ANSWER}")

# --- Search Parameters ---
VECTOR_SIMILARITY_THRESHOLD = 0.60
VECTOR_MATCH_COUNT = 3

# --- Initialize Clients ---
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("Supabase client initialized.")
except Exception as e: print(f"Error initializing Supabase client: {e}"); exit()
try:
    st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Sentence Transformer model ({EMBEDDING_MODEL_NAME}) loaded.")
    test_emb = st_model.encode("test")
    if len(test_emb) != EMBEDDING_DIMENSIONS: raise ValueError("Embedding dimension mismatch")
except Exception as e: print(f"Error loading Sentence Transformer model: {e}"); exit()
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("Groq client initialized.")
except Exception as e: print(f"Error initializing Groq client: {e}"); exit()

# --- Helper Functions ---
def get_query_embedding(text):
    if not text: return None
    try: return st_model.encode(text).tolist()
    except Exception as e: print(f"    Error generating query embedding: {e}"); return None

def find_relevant_markdown_chunks(query_embedding):
    if not query_embedding: return []
    try:
        response = supabase.rpc(RPC_FUNCTION_NAME, {
            'query_embedding': query_embedding,
            'match_threshold': VECTOR_SIMILARITY_THRESHOLD,
            'match_count': VECTOR_MATCH_COUNT
        }).execute()
        return response.data if response.data else []
    except Exception as e: print(f"    Error calling RPC '{RPC_FUNCTION_NAME}': {e}"); return []

# ----- Refined Text-to-SQL Generation Function -----

# ----- Refined Text-to-SQL Generation Function -----
def generate_sql_from_query(user_query, table_schema):
    """Uses Groq LLM with refined prompt and examples to generate SQL, attempting broad keyword matching."""
    print("    Attempting Text-to-SQL generation...")
    # --- Refined Prompt ---
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
            ], model=GROQ_MODEL_FOR_SQL, temperature=0.1, # Keep temperature low for consistency
               max_tokens=1000 # Increase max_tokens slightly as queries might get long
        )
        if not response.choices or not response.choices[0].message: return None
        generated_sql = response.choices[0].message.content.strip()

        if generated_sql == "NO_SQL":
            print("    LLM determined no SQL query is applicable.")
            return None
        # Reuse the robust SQL check from previous response
        if generated_sql.upper().startswith("SELECT") and generated_sql.endswith(';'):
            forbidden = ["UPDATE", "DELETE", "INSERT", "DROP", "TRUNCATE", "ALTER", "CREATE", "EXECUTE", "GRANT", "REVOKE"]
            upper_sql = generated_sql.upper()
            if any(k in upper_sql for k in forbidden):
                print(f"    WARNING: Generated SQL contains forbidden keyword. Discarding: {generated_sql}")
                return None
            # Check if the target table name appears after FROM
            # Regex needs to handle potential schema qualification like "public"."Leoni_attributes"
            # Making it simpler: check if table name (quoted or unquoted) is present after FROM
            table_name_pattern = r'FROM\s+(?:[\w]+\.)?("?' + ATTRIBUTE_TABLE_NAME + r'"?)'
            from_clause_match = re.search(table_name_pattern, generated_sql, re.IGNORECASE)
            if not from_clause_match:
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

# The rest of your script (imports, configuration, find_relevant_attributes_with_sql, main loop etc.) remains the same.
# Make sure to replace the old `generate_sql_from_query` function with this updated version.
# ----- SQL Execution Function (Client-Side OR/AND Handling) -----
def find_relevant_attributes_with_sql(generated_sql):
    """
    Executes SQL using Supabase client filters.
    Attempts to handle simple OR conditions using .or_().
    Falls back to implicit AND for other cases.
    WARNING: Complex WHERE clauses (mixed AND/OR, parentheses) might be executed incorrectly.
    """
    if not generated_sql: return []
    try:
        print(f"    Attempting to execute generated SQL on '{ATTRIBUTE_TABLE_NAME}' (using client-side filters)...")
        print(f"      SQL Generated by LLM: {generated_sql}")

        select_clause = "*"
        sel_match = re.match(r"SELECT\s+(.*?)\s+FROM", generated_sql, re.I | re.S)
        if sel_match:
            select_clause = sel_match.group(1).strip()
            # Ensure "Number" is always selected if not selecting * (for identification)
            if select_clause != "*" and '"Number"' not in select_clause and 'Number' not in select_clause :
                 # Avoid adding "Number" if DISTINCT is used and Number wasn't requested
                 is_distinct = re.match(r"DISTINCT\s+", select_clause, re.I)
                 if not is_distinct:
                    select_clause += ', "Number"'
            # Strip potential trailing comma if added unnecessarily (e.g. SELECT DISTINCT "Col1", "Number")
            select_clause = select_clause.strip().rstrip(',')


        query = supabase.table(ATTRIBUTE_TABLE_NAME).select(select_clause)

        # --- WHERE Clause Parsing ---
        where_clause_full = ""
        where_match = re.search(r'WHERE\s+(.*?)(?:ORDER BY|LIMIT|$)', generated_sql, re.I | re.S)
        if where_match:
            where_clause_full = where_match.group(1).strip().rstrip(';')
            print(f"      Extracted WHERE (approx): {where_clause_full}")

        # Pattern to extract simple Column OP 'Value' conditions
        # Handles quoted/unquoted columns, common operators
        filter_pattern = re.compile(
            # r'("?[\w\s\[\]%.°\-]+"?)\s*(ILIKE|LIKE|=)\s*\'(.*?)\'', # Original - might miss some cases
            r'("?[\w\s%.°\[\]\-]+"?)\s+(ILIKE|LIKE|=|<=?|>=?)\s*\'?([^ \']+)\'?', # Improved: handles operators like >, <, >=, <= and potentially unquoted numeric values
            re.IGNORECASE
        )

        # --- Extract all potential conditions ---
        extracted_conditions = []
        for match in filter_pattern.finditer(where_clause_full):
            # Remove quotes for easier handling in supabase-py, which adds them back if needed
            col = match.group(1).strip().replace('"', '')
            op = match.group(2).upper()
            val = match.group(3).strip() # Value might be numeric/unquoted
            extracted_conditions.append({"col": col, "op": op, "val": val})

        conditions_applied_count = 0
        applied_filter_summary = []

        # --- Decide between OR and AND logic (Simplistic Check) ---
        # Assume OR logic if " OR " is present AND " AND " is NOT present at the top level
        # WARNING: This is a fragile heuristic and fails on mixed/nested logic
        is_likely_or_logic = extracted_conditions and " OR " in where_clause_full.upper() and " AND " not in where_clause_full.upper()

        if is_likely_or_logic:
            print("      Attempting to apply filters using OR logic (client-side .or_())")
            or_filter_parts = []
            valid_or_parse = True
            for cond in extracted_conditions:
                col, op, val = cond["col"], cond["op"], cond["val"]
                supabase_op = None
                if op == 'ILIKE': supabase_op = 'ilike'
                elif op == 'LIKE': supabase_op = 'like'
                elif op == '=': supabase_op = 'eq'
                elif op == '>': supabase_op = 'gt'
                elif op == '<': supabase_op = 'lt'
                elif op == '>=': supabase_op = 'gte'
                elif op == '<=': supabase_op = 'lte'
                # Add other operators as needed (e.g., 'neq', 'in', 'is')

                if supabase_op:
                    # Format for .or_(): "column.operator.value"
                    # Ensure value doesn't contain commas that break .or_ parsing
                    safe_val = str(val).replace(',', '%2C') # URL encode comma just in case
                    or_filter_parts.append(f'{col}.{supabase_op}.{safe_val}')
                    applied_filter_summary.append(f'`{col}` {op} \'{val}\'')
                    conditions_applied_count += 1
                else:
                    print(f"      WARNING: Unsupported operator '{op}' encountered in OR clause for column '{col}'. Cannot reliably use .or_().")
                    valid_or_parse = False
                    break # Stop trying to build the OR string

            if valid_or_parse and or_filter_parts:
                or_string = ",".join(or_filter_parts)
                print(f"      Applying .or_() filter: {or_string}")
                query = query.or_(or_string)
            else:
                 print("      WARNING: Could not reliably parse conditions for .or_(). Falling back to NO filtering for this clause.")
                 # Reset counts as we didn't apply the filter
                 conditions_applied_count = 0
                 applied_filter_summary = []
                 # We can't fall back to AND here as the SQL intended OR. Doing nothing is safer than wrong results.

        elif extracted_conditions:
            # --- Apply filters sequentially (Implicit AND logic) ---
            print("      Applying filters using sequential (implicit AND) logic.")
            for cond in extracted_conditions:
                col, op, val = cond["col"], cond["op"], cond["val"]
                applied_this_filter = False
                try:
                    if op == 'ILIKE': query = query.ilike(col, val); applied_this_filter = True
                    elif op == 'LIKE': query = query.like(col, val); applied_this_filter = True
                    elif op == '=': query = query.eq(col, val); applied_this_filter = True
                    elif op == '>': query = query.gt(col, val); applied_this_filter = True
                    elif op == '<': query = query.lt(col, val); applied_this_filter = True
                    elif op == '>=': query = query.gte(col, val); applied_this_filter = True
                    elif op == '<=': query = query.lte(col, val); applied_this_filter = True
                    # Add other operators as needed
                    else:
                        print(f"      WARNING: Unsupported operator '{op}' for column '{col}' in AND logic.")

                    if applied_this_filter:
                        applied_filter_summary.append(f'`{col}` {op} \'{val}\'')
                        conditions_applied_count += 1
                except Exception as filter_err:
                     print(f"      ERROR applying filter: `{col}` {op} '{val}'. Error: {filter_err}")


        if conditions_applied_count > 0:
            print(f"      Applied {conditions_applied_count} filter condition(s): " + ", ".join(applied_filter_summary))
        elif where_clause_full and not extracted_conditions:
            print("      WARNING: WHERE clause found but no conditions could be parsed by the filter pattern.")
        elif not where_clause_full:
            print("      No WHERE clause detected.")
        # --- End WHERE Clause Parsing ---

        # --- LIMIT Clause Parsing ---
        final_limit_to_apply = 10 # Default
        limit_match = re.search(r'LIMIT\s*(\d+)', generated_sql, re.I | re.S)
        if limit_match:
            final_limit_to_apply = int(limit_match.group(1))
            print(f"      Limit parsed from SQL: {final_limit_to_apply}")
        else:
            print(f"      No LIMIT clause found in SQL, using default: {final_limit_to_apply}")

        query = query.limit(final_limit_to_apply)
        # --- End LIMIT Clause Logic ---

        print(f"      Executing Client-Side Query (approximated): SELECT {select_clause} FROM {ATTRIBUTE_TABLE_NAME} ... [LIMIT {final_limit_to_apply}]")
        response = query.execute()

        if response.data:
            print(f"      Client query returned {len(response.data)} row(s).")
            print("      Retrieved Data (first 3 rows):")
            for i, row_data in enumerate(response.data[:3]):
                 print(f"        Row {i+1}: {json.dumps(row_data, indent=2)}")
            if len(response.data) > 3: print(f"        ... and {len(response.data)-3} more rows.")
            return response.data
        else:
            print("      Client query returned no matching rows.")
            return []
    except Exception as e:
        print(f"    Error executing client-side Supabase query on '{ATTRIBUTE_TABLE_NAME}': {e}")
        import traceback # Keep for debugging complex issues
        traceback.print_exc()
        return []


# --- format_context (No changes needed) ---
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
            # Ensure complex keys (like those with spaces or special chars) are handled if they come from DB
            row_str_parts = []
            for key, value in row.items():
                if value is not None:
                    # Simple formatting for display
                    row_str_parts.append(f"  {key}: {json.dumps(value)}") # Use json.dumps for cleaner output of various types
            context_str += "\n".join(row_str_parts)
            context_str += "\n---\n\n"
    if not md_present and not attr_present:
        return "No relevant information found in the knowledge base (documents or attributes)."
    return context_str.strip()

# --- get_groq_chat_response (No changes needed) ---
def get_groq_chat_response(prompt, context_provided=True):
    if context_provided:
        system_message = "You are a helpful assistant knowledgeable about LEOparts standards and attributes. Answer the user's question based *only* on the provided context from the Standards Document and/or the Attributes Table. The Attributes Table context shows rows retrieved based on the user's query; assume these rows accurately reflect the query's conditions as interpreted by the client-side filters. Synthesize information from both sources if relevant and available. Be concise. If listing items, like part numbers, list them clearly."
    else:
        system_message = "You are a helpful assistant knowledgeable about LEOparts standards and attributes. You were unable to find relevant information in the knowledge base (documents or attributes) to answer the user's question. State clearly that the information is not available in the provided materials. Do not make up information or answer from general knowledge."
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
            model=GROQ_MODEL_FOR_ANSWER, temperature=0.1, stream=False )
        return response.choices[0].message.content
    except Exception as e: print(f"    Error calling Groq API: {e}"); return "Error contacting LLM."


# --- Main Chat Loop ---
print("\n--- LEOparts Standards & Attributes Chatbot (Text-to-SQL Enabled) ---")

# NOTE: Providing the schema helps the LLM generate correct SQL, especially with quoting
leoni_attributes_schema_for_main_loop = """(id: bigint, Number: text, Name: text, "Object Type Indicator": text, Context: text, Version: text, State: text, "Last Modified": timestamp with time zone, "Created On": timestamp with time zone, "Sourcing Status": text, "Material Filling": text, "Material Name": text, "Max. Working Temperature [°C]": numeric, "Min. Working Temperature [°C]": numeric, Colour: text, "Contact Systems": text, Gender: text, "Housing Seal": text, "HV Qualified": text, "Length [mm]": numeric, "Mechanical Coding": text, "Number Of Cavities": numeric, "Number Of Rows": numeric, "Pre-assembled": text, Sealing: text, "Sealing Class": text, "Terminal Position Assurance": text, "Type Of Connector": text, "Width [mm]": numeric, "Wire Seal": text, "Connector Position Assurance": text, "Colour Coding": text, "Set/Kit": text, "Name Of Closed Cavities": text, "Pull-To-Seat": text, "Height [mm]": numeric, Classification: text)"""

while True:
    user_query = input("\nYour Question: ")
    if user_query.lower() == 'quit': break
    if not user_query.strip(): continue

    relevant_markdown_chunks = []
    relevant_attribute_rows = []
    context_was_found = False
    generated_sql = None

    # 1. Attempt Text-to-SQL generation
    generated_sql = generate_sql_from_query(user_query, leoni_attributes_schema_for_main_loop)

    # 2. Execute SQL (using client-side filters)
    if generated_sql:
        relevant_attribute_rows = find_relevant_attributes_with_sql(generated_sql)
        if relevant_attribute_rows: context_was_found = True
    else:
        print(" -> Text-to-SQL generation failed or not applicable.")

    # 3. Perform Vector Search (can be conditional)
    run_vector_search = True # Default: always run it for descriptive context
    # Example: Potentially skip if SQL found *exactly* one specific item and the query was precise
    # if generated_sql and len(relevant_attribute_rows) == 1 and "what is part number" in user_query.lower():
       # run_vector_search = False

    if run_vector_search:
        print(" -> Generating query embedding for descriptive search...")
        query_embedding = get_query_embedding(user_query)
        if query_embedding:
            print(f" -> Searching '{MARKDOWN_TABLE_NAME}' (Vector Search)...")
            relevant_markdown_chunks = find_relevant_markdown_chunks(query_embedding)
            if relevant_markdown_chunks: context_was_found = True
        else:
            print("Error: Could not generate embedding. Skipping vector search.")

    # 4. Prepare Context
    context_str = format_context(relevant_markdown_chunks, relevant_attribute_rows)
    if not context_was_found: print(" -> No relevant information found from either source.")
    else: print(f" -> Found {len(relevant_markdown_chunks)} doc chunk(s) and {len(relevant_attribute_rows)} attribute row(s).")

    # 5. Generate Response
    print(" -> Generating response with Groq...")
    prompt_for_llm = f"""Context:
{context_str}

User Question: {user_query}

Answer the user question based *only* on the provided context."""

    llm_response = get_groq_chat_response(prompt_for_llm, context_provided=context_was_found)
    print("\nAssistant Response:")
    print(llm_response)