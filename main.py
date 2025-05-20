"""
LEOparts Standards & Attributes Chatbot – core engine
Adapted for Streamlit Community Cloud.

• Reads secrets from st.secrets (cloud) or environment/.env (local)
• Generates PostgreSQL queries with Groq (Qwen-32B)
• Falls back to vector search over markdown chunks
• Crafts the final answer with Groq, using only retrieved context
"""

# ────────────────────────────────────────────────────────────────────────────
# Imports & secrets
# ────────────────────────────────────────────────────────────────────────────
import os, re, ast, json
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()                                   # local .env convenience

try:
    import streamlit as st                      # available in cloud & local
    _sx = st.secrets
except ModuleNotFoundError:                     # CLI / unit-tests
    _sx = {}

def _get_secret(name: str) -> str | None:
    return _sx.get(name) or os.getenv(name)

SUPABASE_URL         = _get_secret("SUPABASE_URL")
SUPABASE_SERVICE_KEY = _get_secret("SUPABASE_SERVICE_KEY")
GROQ_API_KEY         = _get_secret("GROQ_API_KEY")
DEBUG_FLAG           = (_get_secret("DEBUG") or "false").lower() == "true"

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY]):
    raise RuntimeError(
        "🔑  Missing SUPABASE_URL / SUPABASE_SERVICE_KEY / GROQ_API_KEY. "
        "Add them in Streamlit Cloud → Settings → Secrets."
    )

VERBOSE = DEBUG_FLAG

# ────────────────────────────────────────────────────────────────────────────
# Heavy libraries (loaded once)
# ────────────────────────────────────────────────────────────────────────────
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from groq import Groq

def _init_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def _init_st_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def _init_groq():
    return Groq(api_key=GROQ_API_KEY)

supabase: Client                 = _init_supabase()
st_model: SentenceTransformer    = _init_st_model()
groq_client: Groq                = _init_groq()

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────
MARKDOWN_TABLE          = "markdown_chunks"
ATTRIBUTE_TABLE         = "Leoni_attributes"
RPC_VECTOR_SEARCH       = "match_markdown_chunks"
RPC_READONLY_SQL        = "execute_readonly_sql"

EMBEDDING_DIM           = 384
SIM_THRESHOLD           = 0.40
SIM_MATCH_COUNT         = 3

GROQ_MODEL_SQL          = "qwen-qwq-32b"
GROQ_MODEL_ANSWER       = "qwen-qwq-32b"

# → real schema text (used in prompt)
LEONI_SCHEMA = """(
id: bigint, Number: text, Name: text, "Object Type Indicator": text, Context: text,
Version: text, State: text, "Last Modified": timestamp with time zone,
"Created On": timestamp with time zone, "Sourcing Status": text,
"Material Filling": text, "Material Name": text,
"Max. Working Temperature [°C]": numeric, "Min. Working Temperature [°C]": numeric,
Colour: text, "Contact Systems": text, Gender: text, "Housing Seal": text,
"HV Qualified": text, "Length [mm]": numeric, "Mechanical Coding": text,
"Number Of Cavities": numeric, "Number Of Rows": numeric, "Pre-assembled": text,
Sealing: text, "Sealing Class": text, "Terminal Position Assurance": text,
"Type Of Connector": text, "Width [mm]": numeric, "Wire Seal": text,
"Connector Position Assurance": text, "Colour Coding": text, "Set/Kit": text,
"Name Of Closed Cavities": text, "Pull-To-Seat": text, "Height [mm]": numeric,
Classification: text)"""

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def strip_think_tags(text: str) -> str:
    return re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>', "", text,
                  flags=re.I | re.S).strip()

def _normalise_chunk(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict) and "content" in row:
        base = row
    else:
        if isinstance(row, dict) and len(row) == 1:
            row = next(iter(row.values()))
        if isinstance(row, str):
            try:
                row = json.loads(row)
            except json.JSONDecodeError:
                try:
                    row = ast.literal_eval(row)
                except Exception:
                    row = {"content": row}
        base = row if isinstance(row, dict) else {"content": str(row)}
    base.setdefault("filename", "Unknown")
    base.setdefault("similarity", None)
    return base

def get_query_embedding(text: str) -> list[float]:
    return st_model.encode(text).tolist()

def find_md_chunks(embed: list[float]) -> List[Dict]:
    resp = supabase.rpc(
        RPC_VECTOR_SEARCH,
        {
            "query_embedding": embed,
            "match_threshold": SIM_THRESHOLD,
            "match_count": SIM_MATCH_COUNT,
        },
    ).execute()
    rows = resp.data or []
    if VERBOSE:
        print("Vector-search rows:", rows[:3])
    return [_normalise_chunk(r) for r in rows]

# ────────────────────────────────────────────────────────────────────────────
# Text-to-SQL
# ────────────────────────────────────────────────────────────────────────────
_SQL_PROMPT_TEMPLATE = f"""Your task is to convert natural-language questions into
robust PostgreSQL **SELECT** queries for the "{ATTRIBUTE_TABLE}" table.

**Strict rules**
1. *Output only SQL or the exact word NO_SQL.* No commentary.
2. Only query "{ATTRIBUTE_TABLE}".
3. Quote columns only when needed. Table schema: {LEONI_SCHEMA}
4. SELECT required columns; use `SELECT *` only for exact part-number lookups.
5. **Keyword robustness:**  
   – Identify descriptive keyword(s) (colour, material, etc.).  
   – Generate abbreviations, alt spellings, typos, different casing.  
   – Search every variation across relevant text columns with `ILIKE '%var%'`.  
   – Combine everything with OR.  
6. `LIMIT 3` for specific numbers; `LIMIT 10-20` for broad searches.  
7. Return NO_SQL for questions outside the table’s scope.

### Examples
User: *What is part number P00001636?*  
SQL: `SELECT * FROM "{ATTRIBUTE_TABLE}" WHERE "Number" = 'P00001636' LIMIT 3;`

User: *List parts that are black*  
SQL… (full robust example omitted here for brevity)

User Question: "{{user_query}}"
SQL:
"""

def generate_sql(user_query: str) -> str | None:
    prompt = _SQL_PROMPT_TEMPLATE.replace("{{user_query}}", user_query)
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL_SQL,
            temperature=0.1,
            max_tokens=4096,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Text-to-SQL assistant.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        sql = strip_think_tags(resp.choices[0].message.content)
    except Exception as e:
        if VERBOSE:
            print("Groq SQL error:", e)
        return None

    if sql == "NO_SQL":
        return None
    if not (sql.upper().startswith("SELECT") and sql.rstrip().endswith(";")):
        if VERBOSE:
            print("Invalid SQL returned:", sql)
        return None
    if re.search(
        r"\b(UPDATE|DELETE|INSERT|DROP|ALTER|TRUNCATE|GRANT|REVOKE)\b",
        sql,
        flags=re.I,
    ):
        return None
    if not re.search(
        rf'FROM\s+(?:\w+\.)?"?{ATTRIBUTE_TABLE}"?', sql, flags=re.I
    ):
        return None
    if VERBOSE:
        print("Generated SQL:", sql)
    return sql

def find_attr_rows(sql: str | None) -> List[Dict]:
    if not sql:
        return []
    try:
        res = supabase.rpc(
            RPC_READONLY_SQL, {"q": sql.rstrip(";")}
        ).execute()
        rows = res.data or []
        if VERBOSE:
            print("SQL rows:", rows[:3])
        if rows and isinstance(rows[0], dict):
            return rows
        first_key = next(iter(rows[0].keys()), None) if rows else None
        return [json.loads(r[first_key]) for r in rows] if first_key else []
    except Exception as e:
        if VERBOSE:
            print("execute_readonly_sql error:", e)
        return []

# ────────────────────────────────────────────────────────────────────────────
# Context & answer
# ────────────────────────────────────────────────────────────────────────────
def _format_context(md_chunks: List[Dict], attr_rows: List[Dict]) -> str:
    out: list[str] = []
    if md_chunks:
        out.append("Context from LEOparts Standards Document:\n")
        for i, c in enumerate(md_chunks, 1):
            out.append(
                f"--- Document Chunk {i} (Source: {c['filename']}) ---\n"
                f"{c['content']}\n"
            )
    if attr_rows:
        out.append("\nContext from Leoni Attributes Table:\n")
        for i, r in enumerate(attr_rows, 1):
            lines = [f"--- Attribute Row {i} ---"]
            lines += [f"  {k}: {json.dumps(v)}" for k, v in r.items() if v]
            out.append("\n".join(lines))
    return "\n".join(out) or "No relevant information."

def _groq_answer(prompt: str, ctx_found: bool) -> str:
    system_msg = (
        "You are a helpful assistant knowledgeable about LEOparts. "
        "Answer ONLY from the provided context."
        if ctx_found
        else "No relevant context available. State this clearly."
    )
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL_ANSWER,
            temperature=0.1,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user",   "content": prompt}],
        )
        return strip_think_tags(resp.choices[0].message.content)
    except Exception as e:
        return f"Error contacting Groq: {e}"

# ────────────────────────────────────────────────────────────────────────────
# Public entry point
# ────────────────────────────────────────────────────────────────────────────
def answer_question(user_q: str) -> str:
    """Single-call helper for Streamlit & CLI."""
    user_q = user_q.strip()
    if not user_q:
        return "Please enter a question."

    sql         = generate_sql(user_q)
    attr_rows   = find_attr_rows(sql)
    md_chunks   = find_md_chunks(get_query_embedding(user_q))
    ctx_found   = bool(attr_rows or md_chunks)
    context     = _format_context(md_chunks, attr_rows)

    prompt = f"Context:\n{context}\n\nUser Question: {user_q}\n"
    return _groq_answer(prompt, ctx_found)

# ────────────────────────────────────────────────────────────────────────────
# CLI fallback
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("LEOparts Chatbot (CLI) – type 'quit' to exit.")
    while True:
        q = input("\nYour Question: ")
        if q.lower() == "quit":
            break
        print("\n>>>", answer_question(q))
