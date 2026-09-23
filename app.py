"""
Northbridge Bank - Credit Risk Natural-Language Query Engine (Streamlit app)

Converted from: Learner_Notebook_Project_3_Credit_Risk_Query_Engine_NGW.ipynb

Pipeline (unchanged from the notebook):
    classify_intent -> (verified template | generate_query) -> validate_query
    -> (retry_generation once, generated route only) -> execute_query
    -> generate_response, with escalation to a human analyst if validation fails.

Files expected next to this app.py:
    credit_risk_portfolio.db   SQLite database (opened read-only)
    test_queries.csv           ground-truth cases for the evaluation tab

Credentials are read from (first match wins):
    1. Streamlit secrets  (.streamlit/secrets.toml or the Streamlit Cloud "Secrets" box)
    2. Environment variables (OPENAI_API_KEY, OPENAI_API_BASE)
    3. config.json        (same format the notebook used; local development only)
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import json                                   # Read and write JSON data
import os                                     # Environment variables and paths
import re                                     # Regular-expression text processing
import sqlite3                                # Connect to and query the SQLite database
import time                                   # Measure turnaround time
from datetime import datetime, timezone       # Audit-trail timestamps
from pathlib import Path                      # Robust file paths

import pandas as pd                           # Data manipulation and analysis
import sqlparse                               # Parse and format SQL queries
import streamlit as st                        # Web UI

from langchain_openai import ChatOpenAI       # Use OpenAI chat models through LangChain
from openai import OpenAI                     # OpenAI Python client

import warnings                               # Manage Python warning messages
warnings.filterwarnings('ignore')             # Suppress warning messages

# Pandas configuration for better readability (same as notebook)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)

# =============================================================================
# 2. PAGE CONFIG + LIGHT STYLING  (st.set_page_config must be the first st.* call)
# =============================================================================
st.set_page_config(
    page_title="Credit Risk Query Engine | Northbridge Bank",
    page_icon="🏦",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1250px; }
      div[data-testid="stMetric"] {
          border: 1px solid rgba(128,128,128,.28);
          border-radius: 6px;
          padding: .55rem .8rem;
      }
      div[data-testid="stMetricValue"] { font-size: 1.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 3. CONFIGURATION, CREDENTIALS, DATABASE PATH
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent

file_name = 'config.json'                                        # Optional local config file (as in the notebook)
db_path = os.environ.get("DB_PATH", 'credit_risk_portfolio.db')  # SQLite database file
test_queries_path = os.environ.get("TEST_QUERIES_PATH", 'test_queries.csv')
AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", 'audit_log.jsonl')


def resolve_path(name):
    """Resolve a file name relative to the folder that contains app.py."""
    p = Path(name)
    return p if p.is_absolute() else BASE_DIR / p


def _read_secret(name):
    """Read a value from Streamlit secrets; return None if secrets are not configured."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or None


def load_credentials():
    """Return (api_key, api_base) from Streamlit secrets, env vars, or config.json."""
    api_key = _read_secret("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = (
        _read_secret("OPENAI_API_BASE")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
    )
    config_file = resolve_path(file_name)
    if (not api_key or not api_base) and config_file.exists():
        try:
            with open(config_file, 'r') as file:                 # Open the config file in read mode
                config = json.load(file)                         # Load the JSON content as a dictionary
                api_key = api_key or config.get("OPENAI_API_KEY")
                api_base = api_base or config.get("OPENAI_API_BASE")
        except (OSError, json.JSONDecodeError):
            pass
    return api_key, api_base


OPENAI_API_KEY, OPENAI_API_BASE = load_credentials()

if not OPENAI_API_KEY:
    st.title("Credit Risk Query Engine")
    st.error(
        "OpenAI credentials were not found. Add `OPENAI_API_KEY` (and `OPENAI_API_BASE` if you use a "
        "custom endpoint) to Streamlit secrets, set them as environment variables, or provide a local "
        "`config.json`. See the setup notes for details."
    )
    st.stop()

# Store API credentials in environment variables (same as notebook)
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE

if not resolve_path(db_path).exists():
    st.title("Credit Risk Query Engine")
    st.error(f"Database file not found: `{resolve_path(db_path)}`. "
             "Place `credit_risk_portfolio.db` next to app.py (or set the DB_PATH environment variable).")
    st.stop()


# =============================================================================
# 4. MODEL INITIALISATION
# =============================================================================
@st.cache_resource(show_spinner=False)
def init_models(_api_key, _api_base):
    """Create the LLM clients once per server process."""
    llm_ = ChatOpenAI(model='gpt-4o-mini', temperature=0)             # Router, SQL generation, narrative
    evaluator_llm_ = ChatOpenAI(model='gpt-4o', temperature=0)        # Relevance judge in the validation gate
    client_ = OpenAI()                                                # Raw OpenAI client for llm_response()
    return llm_, evaluator_llm_, client_


llm, evaluator_llm, client = init_models(OPENAI_API_KEY, OPENAI_API_BASE)


# ── OpenAI Chat Completions ───────────────────────────────────────────────────
def llm_response(system: str, user: str, temperature: float = 0.1) -> str:
    """Single-turn chat completion. Returns the assistant content string."""
    # Create a completion using the initialized client and global MODEL variable
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},  # Sets the AI persona/rules
            {"role": "user",   "content": user},    # The specific user prompt
        ],
    )
    # Extract and return the text content from the first response choice
    return response.choices[0].message.content


# ── OpenAI Chat Completions ───────────────────────────────────────────────────
def judge_llm_response(system: str, user: str, temperature: float = 0.1) -> str:
    """Single-turn chat completion. Returns the assistant content string."""
    # Create a completion using the initialized client and global MODEL variable
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},  # Sets the AI persona/rules
            {"role": "user",   "content": user},    # The specific user prompt
        ],
    )
    # Extract and return the text content from the first response choice
    return response.choices[0].message.content


# =============================================================================
# 5. DATABASE (READ-ONLY) + SCHEMA CONTEXT + GROUND TRUTH
# =============================================================================
def get_db_connection():
    """
    Read-only connection to the SQLite database (URI mode with ?mode=ro prevents any write).
    A fresh short-lived connection is opened per pipeline run because Streamlit executes
    scripts on different threads and sqlite3 connections are bound to their creating thread.
    """
    return sqlite3.connect(f"{resolve_path(db_path).as_uri()}?mode=ro", uri=True)


database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


@st.cache_data(show_spinner=False)
def load_ground_truth(path):
    """Load test_queries.csv (ground-truth test cases). Returns None if the file is missing."""
    p = resolve_path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


ground_truth = load_ground_truth(test_queries_path)


# =============================================================================
# 6. VERIFIED QUERY TEMPLATE LIBRARY
# =============================================================================

# ── Verified Query 1: Sector-wise Outstanding and NPA Breakdown ───────────────
sql_1 = {
    'VQ1': {
        'description': 'Calculated total outstanding exposure and NPA exposure for each sector',
        'sql': """SELECT
    sm.sector_name,
    ROUND(SUM(lm.total_outstanding) / 1000000.0, 2) AS total_outstanding_mm,
    ROUND(
        SUM(
            CASE
                WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                THEN lm.total_outstanding
                ELSE 0
            END
        ) / 1000000.0, 2
    ) AS npa_exposure_mm
FROM loan_master lm
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
GROUP BY sm.sector_name
ORDER BY total_outstanding_mm DESC;"""}
}

# ── Verified Query 2: Portfolio Outstanding by Loan Category ──────────────────
sql_2 = {
    'VQ2': {
        'description': 'Total outstanding exposure and loan count for each loan category',
        'sql': """SELECT
    loan_category,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mm
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_mm DESC;"""}
}

# ── Verified Query 3: IFRS 9 Stage-wise ECL Summary ───────────────────────────
sql_3 = {
    'VQ3': {
        'description': 'Summarized loan exposure and expected credit loss by IFRS 9 stage for the latest reporting quarter',
        'sql': """SELECT
    ifrs9_stage,
    COUNT(*) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_mm,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mm
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage;"""}
}

# ── Verified Query 4: Provision Coverage Ratio by Sector ──────────────────────
sql_4 = {
    'VQ4': {
        'description': 'Average provision coverage ratio for each sector in the last reporting quarter',
        'sql':
"""SELECT
    sm.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 4) AS avg_provision_coverage_ratio
FROM provisioning p
JOIN loan_master lm
    ON p.loan_account_number = lm.loan_account_number
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY sm.sector_name
ORDER BY avg_provision_coverage_ratio DESC;"""}
}

# ── Verified Query 5: Top 10 Loan Exposures ───────────────────────────────────
sql_5 = {
    'VQ5': {
        'description': 'The ten individual loans with the highest outstanding exposure',
        'sql':
"""SELECT
lm.borrower_name,
    sm.sector_name AS sector,
    ROUND(lm.total_outstanding / 1000000.0, 2) AS outstanding_exposure_mm,
    lm.asset_classification
FROM loan_master lm
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
ORDER BY outstanding_exposure_mm DESC
LIMIT 10;"""}
}

# ── Verified Query 6: Top 5 Business Group Exposures ──────────────────────────
sql_6 = {
    'VQ6': {
        'description': 'The five business groups with the highest total outstanding exposure',
        'sql': """SELECT
    group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_exposure_mm
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY total_exposure_mm DESC
LIMIT 5;"""}
}

# ── Verified Query 7: All Overdue Loan Accounts ───────────────────────────────
sql_7 = {
    'VQ7': {
        'description': 'The loans that are currently overdue, with their delinquency information',
        'sql': """SELECT
    lm.loan_account_number,
    lm.borrower_name,
    sm.sector_name AS sector,
    ROUND(lm.total_outstanding / 1000000.0, 2) AS outstanding_exposure_mm,
    lm.days_past_due,
    lm.asset_classification
FROM loan_master lm
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
WHERE lm.days_past_due > 0
ORDER BY lm.days_past_due DESC;"""}
}

# ── Verified Query 8: DPD Bucket Distribution ─────────────────────────────────
sql_8 = {
    'VQ8': {
        'description': 'How the loans and outstanding exposure are distributed across different days past due buckets',
        'sql': """SELECT
    dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mm
FROM (
    SELECT
        total_outstanding,
        CASE
            WHEN days_past_due = 0 THEN '0 (Current)'
            WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
            WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
            WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
            WHEN days_past_due > 90 THEN '90+'
        END AS dpd_bucket,
        CASE
            WHEN days_past_due = 0 THEN 1
            WHEN days_past_due BETWEEN 1 AND 30 THEN 2
            WHEN days_past_due BETWEEN 31 AND 60 THEN 3
            WHEN days_past_due BETWEEN 61 AND 90 THEN 4
            WHEN days_past_due > 90 THEN 5
        END AS bucket_order
    FROM loan_master
) bucketed
GROUP BY dpd_bucket, bucket_order
ORDER BY bucket_order;"""}
}

# ── Verified Query 9: Latest Rating Downgrades ────────────────────────────────
sql_9 = {
    'VQ9': {
        'description': 'The borrowers whose internal credit rating was downgraded in the latest rating cycle.',
        'sql': """SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
    AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC;"""}
}

# ── Verified Query 10: ECL Trend Across Reporting Quarters ────────────────────
sql_10 = {
    'VQ10': {
        'description': 'How total expected credit loss has changed across the reporting quarters',
        'sql': """SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mm
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date ASC;"""}
}

# ── Verified Query Library ────────────────────────────────────────────────────
verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1['VQ1']['sql']
    },

    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2['VQ2']['sql']
    },

    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3['VQ3']['sql']
    },

    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4['VQ4']['sql']
    },

    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5['VQ5']['sql']
    },

    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6['VQ6']['sql']
    },

    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7['VQ7']['sql']
    },

    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8['VQ8']['sql']
    },

    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle.',
        'sql': sql_9['VQ9']['sql']
    },

    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10['VQ10']['sql']
    }
}


# =============================================================================
# 7. TOOL DEFINITIONS
# =============================================================================

# ── Intent Classification Tool ────────────────────────────────────────────────
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""

### ROLE
You are a query router for a a bank loan operations analytics system. Your job is to decide whether a business user's question can be answered by one of the pre-approved query templates, or whether it needs fresh SQL generation.

### INPUT
User Question:
{user_question}

Available Verified Query Templates:
{library_descriptions}

### INSTRUCTIONS
1. Read the user question carefully and identify the analytical intent.
2. Compare the intent against each template description.
3. Match on semantic meaning, not exact wording. For example,
4. If a template genuinely answers the question, return that template ID.
5. If no template covers the question, return null for the query_id and set the route to generated.
6. Be careful about shape of answer: a question asking for row-level detail (e.g., 'show me the shipments') should NOT match a template that returns an aggregate count.

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# ── Query Generation Tool ─────────────────────────────────────────────────────
def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""

### ROLE
You are a senior SQL developer specializing in bank loan operations analytics on a SQLite database.

### INPUT
User Question:
{user_question}

Database Schema (single source of truth):
{schema_context}

### INSTRUCTIONS
1. Write a single SQL query that answers the user question using only the provided schema.
2. The query must be read-only. Use SELECT (or WITH ... SELECT). Never use DROP, DELETE, UPDATE, INSERT, ALTER, or TRUNCATE.
3. Use only the tables and columns listed in the schema. Do not invent columns.
5. Ensure the query is SQLite compatible.
6. In SQLite, never subtract DATE() or date columns directly (e.g. DATE(a)-DATE(b)) — it silently returns 0; always use julianday(a)-julianday(b) for day differences.
7. Alias every numeric column with a suffix that states its unit, so the result is self-describing. Use _usd for dollar amounts, _pct or _percent for percentages, _count for counts, _hours for hour values, and _days for day values. Avoid bare aliases like "value", "amount", or "total".

### OUTPUT
Return ONLY the SQL query, with no markdown code blocks, no comments, and no explanation.
"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


# ── Query Validation Tool ─────────────────────────────────────────────────────
def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select','from','where','and','or','group','by','order','having','limit','join','on','as','case',
                    'when','then','else','end','sum','count','avg','min','max','round','desc','asc','left','right',
                    'inner','outer','distinct','null','is','not','in','like','with','union','all','between','coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""

### ROLE
You are a senior data validator. Your job is to check whether a SQL query
correctly answers a business user's question about supply chain operations.

### CONTEXT
{track_context}

### INPUT
User Question: {user_question}

Candidate SQL:
{candidate_sql}

### INSTRUCTIONS
Assess whether the SQL genuinely answers what the user asked, considering:
1. Does it query the correct tables and columns?
2. Does it apply the right aggregations and groupings?
3. Does it handle the requested business definitions correctly?
4. Does it resolve named entities correctly?
5. Does it return the right shape of answer?
6. If this is a verified template, do not penalize it for returning
   a broader result set than the question's scope.


### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            # Streamlit-port fix: the notebook appended " LIMIT 0" directly to the SQL text, which fails for
            # every template (trailing ';' -> "only one statement at a time"; VQ5/VQ6 already have a LIMIT).
            # Wrapping each query in a sub-select yields the same column list without those problems.
            expected_body = expected_sql.strip().rstrip(';')
            candidate_body = candidate_sql.strip().rstrip(';')
            expected_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({expected_body}) LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({candidate_body}) LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# ── Retry Generation Tool ─────────────────────────────────────────────────────
def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
### ROLE
You are a senior SQL developer fixing a query that failed validation.

### INPUT
User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

### INSTRUCTIONS
1. Fix only the specific issue identified by the validation error.
2. Preserve the original intent of the query.
3. The revised SQL must be read-only SELECT (or WITH ... SELECT).
4. Use only tables and columns from the schema.
5. Ensure the query is SQLite compatible.

### OUTPUT
Return ONLY the corrected SQL, with no markdown code blocks, no comments, and no explanation.

"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# ── Query Execution Tool ──────────────────────────────────────────────────────
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# ── Response Generation Tool ──────────────────────────────────────────────────
def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""

### ROLE
You are a supply chain operations analyst writing a concise business response for a fulfillment or inventory question.

### INPUT
User Question: {user_question}

Query Result Data:
{dataframe.to_string()}

### INSTRUCTIONS
1. Answer the user's specific question directly. Do not dump the entire table.
2. If the user asked about a specific region, category,loan type, borrower type, group name, state or product type highlight only those rows.
3. Provide context from other rows only when it adds value (for example, ranking or comparison).
4. State exact numbers from the data. Do not round beyond what is shown.
5. Flag anything notable,
6. Use clear, professional language suitable for a bank operations memo.
7. Keep the response focused. Two to four sentences for simple questions, up to a short paragraph for complex ones.
8. State the unit for every number, inferred from its column name: _usd as "$X", _pct or _percent as "X%", _count as a plain count, _hours as "X hours". Never state a bare number when the source column implies a unit.

### OUTPUT
Return ONLY the natural language response text, with no markdown headers or bullet points unless truly needed.
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# =============================================================================
# 8. PIPELINE ORCHESTRATION
# =============================================================================
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True, trace=None):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, records intermediate pipeline stages.
    - trace (list, optional): Streamlit adaptation. When a list is supplied the stage messages
                              are appended to it (shown in the UI) instead of being printed.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None,
        'warnings': []
    }

    def _emit(message):
        # Streamlit adaptation of the notebook's print() calls
        if not verbose:
            return
        if trace is not None:
            trace.append(message)
        else:
            print(message)

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    if verbose:
        _emit(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
        _emit(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    if verbose:
        _emit(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    if verbose:
        _emit(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            _emit(f"    Failed check: {gate.get('failed_check')}")
            _emit(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        if verbose:
            _emit(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        if verbose:
            _emit(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
            if not gate['passed']:
                _emit(f"    Retry failed check: {gate.get('failed_check')}")
                _emit(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        if verbose:
            _emit(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)
    log['warnings'] = exec_result['warnings']

    if verbose:
        _emit(f"[4] Execute: {len(df)} rows returned")
        if exec_result['warnings']:
            _emit(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    if verbose:
        _emit(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# =============================================================================
# 9. STREAMLIT HELPERS: AUDIT TRAIL, RUNNER, RENDERING
# =============================================================================
SAMPLE_QUESTIONS_FALLBACK = [
    "Please show me all of the corporate loan entities",
    "Please show the total amount of outstanding debts across the impaired loans",
    "Please show total average interest rate percentage",
    "Please show the percentage amount of loans across sectors that are secure ordered from highest to lowest",
]

for _key, _default in {'audit_log': [], 'last_run': None, 'test_runs': None, 'question_input': ''}.items():
    st.session_state.setdefault(_key, _default)


def md_safe(text):
    """Escape '$' so Streamlit markdown does not treat dollar amounts as LaTeX."""
    return str(text).replace('$', r'\$')


def format_confidence(conf):
    if isinstance(conf, (int, float)) and not isinstance(conf, bool):
        return f"{conf:.2f}"
    return str(conf) if conf else "n/a"


def format_sql(sql, pretty):
    """Display-only formatting. The audit trail always stores the exact executed SQL."""
    if not pretty or not sql:
        return sql
    try:
        return sqlparse.format(sql, reindent=True, keyword_case='upper')
    except Exception:
        return sql


def persist_audit_record(record):
    """Append the audit record to a JSON-lines file (best effort; ignored if the filesystem is read-only)."""
    try:
        with open(resolve_path(AUDIT_LOG_PATH), 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(record, default=str) + '\n')
        return True
    except OSError:
        return False


def execute_pipeline_for_ui(question, source):
    """Run the pipeline on a fresh read-only connection, time it, and write the audit record."""
    trace = []
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    conn = get_db_connection()
    try:
        result = run_pipeline(question, conn, verified_query_library, database_schema,
                              verbose=True, trace=trace)
    except Exception as exc:                                       # API / parsing / SQL errors
        elapsed = time.perf_counter() - started
        record = {'timestamp_utc': timestamp, 'source': source, 'user_question': question,
                  'elapsed_seconds': round(elapsed, 2), 'error': f"{type(exc).__name__}: {exc}",
                  'trace': trace}
        st.session_state['audit_log'].append(record)
        persist_audit_record(record)
        return {'error': record['error'], 'question': question, 'trace': trace, 'elapsed': elapsed}
    finally:
        conn.close()

    elapsed = time.perf_counter() - started
    df = result['dataframe']
    record = {'timestamp_utc': timestamp, 'source': source, 'elapsed_seconds': round(elapsed, 2),
              **result['log'],
              'columns': list(df.columns) if df is not None else None,
              'trace': trace}
    st.session_state['audit_log'].append(record)
    persist_audit_record(record)
    return {'result': result, 'question': question, 'trace': trace, 'elapsed': elapsed}


def render_run(run, key_prefix, pretty_sql=False, show_trace=True):
    """Render one pipeline run: answer, confidence, SQL used, raw data, validation details, trace."""
    if 'error' in run:
        st.error(f"The pipeline raised an error: {run['error']}")
        if show_trace and run.get('trace'):
            with st.expander("Pipeline trace"):
                st.code("\n".join(run['trace']), language='text')
        return

    result = run['result']
    log = result['log']
    df = result['dataframe']
    gate = log['gate_result'] or {}

    if log['escalated']:
        st.error("This question could not be reliably resolved and has been escalated to a human analyst.")
        st.markdown(md_safe(log['narrative']))
    else:
        st.markdown("##### Answer")
        st.markdown(md_safe(log['narrative']))

    # Headline metrics
    target = 120 if log['route'] == 'verified' else 300
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Route", "Verified template" if log['route'] == 'verified' else "Generated SQL")
    c2.metric("Query ID", log['query_id'] or "-")
    c3.metric("Confidence", format_confidence(log['confidence']))
    c4.metric("Rows returned", "-" if log['row_count'] is None else f"{log['row_count']:,}")
    c5.metric("Response time", f"{run['elapsed']:.1f}s")
    if isinstance(log['confidence'], (int, float)):
        st.progress(max(0.0, min(1.0, float(log['confidence']))))
    st.caption(f"Turnaround target for this route: under {target} seconds.")

    if log['warnings']:
        st.warning("Result checks: " + "; ".join(log['warnings']))

    # SQL used (executed SQL, or the last attempted SQL if escalated)
    sql_shown = log['executed_sql'] or log['candidate_sql']
    with st.expander("SQL used" if not log['escalated'] else "SQL attempted (not executed)", expanded=True):
        st.code(format_sql(sql_shown, pretty_sql), language='sql')

    # Raw data returned
    if df is not None:
        with st.expander("Raw data returned", expanded=True):
            st.dataframe(df, hide_index=True)
            st.download_button("Download result as CSV", df.to_csv(index=False).encode('utf-8'),
                               file_name=f"{key_prefix}_result.csv", mime="text/csv",
                               key=f"{key_prefix}_csv")

    # Routing and validation details
    with st.expander("Routing and validation details"):
        st.markdown(f"**Match reason:** {md_safe(log['match_reason'])}")
        if log['query_id'] in verified_query_library:
            st.markdown(f"**Template:** {log['query_id']} - {verified_query_library[log['query_id']]['description']}")
        st.markdown(f"**Validation passed:** {gate.get('passed')}  \n"
                    f"**Failed check:** {gate.get('failed_check') or 'none'}  \n"
                    f"**Details:** {md_safe(gate.get('details'))}  \n"
                    f"**Relevance confidence:** {format_confidence(gate.get('relevance_confidence'))}  \n"
                    f"**Retry used:** {log['retry_used']}")

    if show_trace and run.get('trace'):
        with st.expander("Pipeline trace"):
            st.code("\n".join(run['trace']), language='text')


def render_evaluation(gt_df, runs):
    """Evaluation against ground truth (same metrics and logic as the notebook)."""
    evaluation_rows = []
    for i, (_, gt) in enumerate(gt_df.iterrows()):
        run = runs[i] if i < len(runs) else {}
        tr = run.get('result') or {'route': None, 'query_id': None, 'confidence': None, 'row_count': None}

        evaluation_rows.append({
            'Test Case': gt['Test Case'],
            'Expected Route': gt['Expected Route'],
            'Actual Route': tr['route'],
            'Route Match': tr['route'] == gt['Expected Route'],
            'Expected Query ID': gt['Expected Query ID'],
            'Actual Query ID': tr['query_id'],
            'Query ID Match': (
                pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
            ) or tr['query_id'] == gt['Expected Query ID'],
            'Confidence': tr['confidence'],
            'Rows Returned': tr['row_count']
        })

    evaluation_df = pd.DataFrame(evaluation_rows)

    path_accuracy = evaluation_df['Route Match'].mean() * 100

    verified = evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
    query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100

    average_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce').mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
    m2.metric("Selected Query Accuracy", "n/a" if pd.isna(query_accuracy) else f"{query_accuracy:.1f}%")
    m3.metric("Average Confidence Score", "n/a" if pd.isna(average_confidence) else f"{average_confidence:.2f}")

    display_df = evaluation_df.copy()
    display_df['Confidence'] = display_df['Confidence'].apply(format_confidence)
    st.dataframe(display_df, hide_index=True)
    st.download_button("Download evaluation summary (CSV)", display_df.to_csv(index=False).encode('utf-8'),
                       file_name="evaluation_summary.csv", mime="text/csv", key="eval_csv")


def set_question(text):
    st.session_state['question_input'] = text


# =============================================================================
# 10. SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("Northbridge Bank")
    st.caption("Credit Risk Analytics - commercial lending portfolio")

    st.markdown("**System**")
    st.markdown(
        f"- Database: read-only connection  \n"
        f"- Verified templates: {len(verified_query_library)}  \n"
        f"- Routing, SQL and narrative model: gpt-4o-mini  \n"
        f"- Validation judge: gpt-4o"
    )

    st.markdown("**Display options**")
    show_trace = st.checkbox("Show pipeline trace", value=True)
    pretty_sql = st.checkbox("Pretty-print SQL (display only)", value=False)

    st.markdown("**Example questions**")
    if ground_truth is not None and 'User Query' in ground_truth.columns:
        examples = [str(q) for q in ground_truth['User Query'].head(5)]
    else:
        examples = SAMPLE_QUESTIONS_FALLBACK
    for i, q in enumerate(examples):
        label = q if len(q) <= 70 else q[:67] + "..."
        st.button(label, key=f"example_{i}", help=q, on_click=set_question, args=(q,))

    with st.expander("Diagnostics"):
        if st.button("Test LLM connection"):
            try:
                st.write("gpt-4o-mini:", llm_response("You are a helpful AI assistant", "Capital of France is"))
                st.write("gpt-4o:", judge_llm_response("You are a helpful AI assistant", "Capital of Italy is"))
            except Exception as exc:
                st.error(f"LLM call failed: {type(exc).__name__}: {exc}")


# =============================================================================
# 11. MAIN PAGE
# =============================================================================
st.title("Credit Risk Query Engine")
st.write(
    "Ask routine commercial lending portfolio questions in plain English. Recurring questions are answered "
    "from pre-approved SQL templates; other questions get freshly generated, read-only SQL that must pass a "
    "validation gate. Every answer shows the SQL used, the raw data and a confidence score, and complex or "
    "unclear requests are escalated to a human analyst."
)

tab_ask, tab_ref, tab_eval, tab_audit = st.tabs(
    ["Ask a question", "Query library and schema", "Test cases and evaluation", "Audit trail"]
)

# ── Tab 1: Ask a question ─────────────────────────────────────────────────────
with tab_ask:
    question = st.text_area(
        "Your question",
        key="question_input",
        height=90,
        placeholder="e.g. What is our total NPA exposure by sector?",
    )
    if st.button("Run query", type="primary"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Routing, validating and running your question..."):
                st.session_state['last_run'] = execute_pipeline_for_ui(question.strip(), 'ask')

    if st.session_state['last_run'] is not None:
        st.divider()
        st.markdown(f"**Question:** {md_safe(st.session_state['last_run']['question'])}")
        render_run(st.session_state['last_run'], 'ask', pretty_sql=pretty_sql, show_trace=show_trace)

# ── Tab 2: Query library and schema ───────────────────────────────────────────
with tab_ref:
    st.markdown("#### Verified query template library")
    st.caption("Pre-approved, tested SQL. The router matches a question to one of these before any SQL is generated.")
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}: {entry['description']}"):
            st.code(format_sql(entry['sql'], pretty_sql), language='sql')

    st.markdown("#### Database schema provided to the LLM")
    with st.expander("View schema context"):
        st.code(database_schema.strip(), language='text')

    st.markdown("#### Table preview (first 5 rows)")
    try:
        _conn = get_db_connection()
        try:
            _tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", _conn)['name'].tolist()
            _selected = st.selectbox("Table", _tables)
            if _selected:
                st.dataframe(pd.read_sql_query(f"SELECT * FROM {_selected} LIMIT 5", _conn), hide_index=True)
        finally:
            _conn.close()
    except Exception as exc:
        st.error(f"Could not read the database: {exc}")

# ── Tab 3: Test cases and evaluation ──────────────────────────────────────────
with tab_eval:
    st.markdown("#### Evaluation against ground truth")
    if ground_truth is None:
        st.warning("`test_queries.csv` was not found next to app.py, so the evaluation tab is unavailable.")
    else:
        st.caption("Each test case runs through the full pipeline. Route, query ID and confidence are compared with the ground truth.")
        st.dataframe(ground_truth, hide_index=True)

        if st.button("Run all test cases", type="primary"):
            runs = []
            progress = st.progress(0.0, text="Starting test cases...")
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                progress.progress(i / len(ground_truth), text=f"Running {gt['Test Case']} ({i + 1} of {len(ground_truth)})...")
                runs.append(execute_pipeline_for_ui(str(gt['User Query']), 'evaluation'))
            progress.empty()
            st.session_state['test_runs'] = runs

        test_runs = st.session_state['test_runs']
        if test_runs:
            render_evaluation(ground_truth, test_runs)

            st.markdown("#### Test case details")
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                if i >= len(test_runs):
                    break
                with st.expander(f"{gt['Test Case']}: {gt['User Query']}"):
                    st.markdown(f"**Expected route:** {gt['Expected Route']}  \n"
                                f"**Expected query ID:** {gt['Expected Query ID'] if not pd.isna(gt['Expected Query ID']) else '-'}")
                    if 'Expected Answer' in ground_truth.columns:
                        st.markdown(f"**Expected answer:** {md_safe(gt['Expected Answer'])}")
                    st.divider()
                    render_run(test_runs[i], f"eval_{i}", pretty_sql=pretty_sql, show_trace=show_trace)

# ── Tab 4: Audit trail ────────────────────────────────────────────────────────
with tab_audit:
    st.markdown("#### Audit trail")
    audit_log = st.session_state['audit_log']
    st.caption(
        "Every pipeline run in this session is recorded here with its route, SQL, validation result and timing. "
        f"Records are also appended to `{AUDIT_LOG_PATH}` on the server when the filesystem is writable."
    )
    if not audit_log:
        st.info("No queries have been run in this session yet.")
    else:
        summary = pd.DataFrame([{
            'Time (UTC)': r.get('timestamp_utc'),
            'Source': r.get('source'),
            'Question': r.get('user_question'),
            'Route': r.get('route'),
            'Query ID': r.get('query_id'),
            'Confidence': format_confidence(r.get('confidence')) if 'error' not in r else 'error',
            'Escalated': r.get('escalated'),
            'Retry used': r.get('retry_used'),
            'Rows': r.get('row_count'),
            'Seconds': r.get('elapsed_seconds'),
        } for r in audit_log])
        st.dataframe(summary, hide_index=True)

        d1, d2 = st.columns(2)
        d1.download_button("Download full audit log (JSON)", json.dumps(audit_log, indent=2, default=str),
                           file_name="audit_log.json", mime="application/json", key="audit_json")
        d2.download_button("Download summary (CSV)", summary.to_csv(index=False).encode('utf-8'),
                           file_name="audit_summary.csv", mime="text/csv", key="audit_csv")

        idx = st.selectbox("Inspect a record", range(len(audit_log)),
                           format_func=lambda i: f"{i + 1}. {audit_log[i].get('user_question', '')[:80]}")
        st.json(audit_log[idx])
