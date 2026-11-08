"""
CreditWise - Loan Intelligence Platform
Developer: Hamza Malik (hamzamalak357@gmail.com)

Run:  streamlit run app.py   (needs Streamlit >= 1.40)
Needs in the same folder:  loan_model.pkl, scaler.pkl
Educational ML-based assessment only. Not an official lending service.

IMPORTANT: Everything marked  [VERIFY]  in the PREPROCESSING CONFIG block was
written WITHOUT access to main.ipynb. Open your notebook and make these match
it exactly (education encoding, category lists / dropped baselines, squared
features, label meaning). Nothing else in the file needs to change.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="CreditWise | Loan Intelligence Platform",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "loan_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
DEVELOPER_EMAIL = "hamzamalak357@gmail.com"
CURRENCY = "₹"  # display label only; change if your dataset uses another unit

# ============================================================================
# PREPROCESSING CONFIG  (single source of truth for feature construction)
# ============================================================================

# Exact 27 training columns, exact order (given by you).
FEATURE_COLUMNS = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Age",
    "Dependents",
    "Existing_Loans",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
    "Education_Level",
    "Employment_Status_Salaried",
    "Employment_Status_Self-employed",
    "Employment_Status_Unemployed",
    "Marital_Status_Single",
    "Loan_Purpose_Car",
    "Loan_Purpose_Education",
    "Loan_Purpose_Home",
    "Loan_Purpose_Personal",
    "Property_Area_Semiurban",
    "Property_Area_Urban",
    "Gender_Male",
    "Employer_Category_Government",
    "Employer_Category_MNC",
    "Employer_Category_Private",
    "Employer_Category_Unemployed",
    "DTI_Ratio_sq",
    "Credit_score_sq",
]

# [VERIFY] Education label encoding used in the notebook.
EDUCATION_MAP = {"Graduate": 1, "Not Graduate": 0}

# [VERIFY] Category choices shown to the user. The dropped baseline category
# (all its dummy columns = 0) is the one with no column in FEATURE_COLUMNS.
CATEGORY_OPTIONS = {
    "Employment_Status": ["Salaried", "Self-employed", "Unemployed", "Contract"],  # baseline: Contract
    "Employer_Category": ["Government", "MNC", "Private", "Unemployed", "Business"],  # baseline: Business
    "Marital_Status": ["Married", "Single"],  # baseline: Married
    "Loan_Purpose": ["Home", "Car", "Education", "Personal", "Business"],  # baseline: Business
    "Property_Area": ["Urban", "Semiurban", "Rural"],  # baseline: Rural
    "Gender": ["Male", "Female"],  # baseline: Female
}

# [VERIFY] Engineered features: assumed plain squares of the raw inputs.
def engineer_dti_sq(dti_ratio: float) -> float:
    return float(dti_ratio) ** 2


def engineer_credit_sq(credit_score: float) -> float:
    return float(credit_score) ** 2


# [VERIFY] Label 1 = loan approved in your target column.
POSITIVE_CLASS = 1

# Your measured test-set metrics (Logistic Regression).
EVAL_METRICS = [
    ("Accuracy", "88%"),
    ("Precision", "78.46%"),
    ("Recall", "83.61%"),
    ("F1 Score", "80.95%"),
]

# ============================================================================
# STYLING  (institutional light theme: white surfaces, one action colour,
# single sans family, restrained bronze accent, dark ink footer)
# ============================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --ink:#12202E; --text:#31373D; --muted:#5F6B76; --line:#D5DCE3;
  --surface:#EEF2F6; --white:#FFFFFF; --blue:#1A7B99; --blue-d:#005075;
  --bronze:#936846; --deep:#0B2A43; --ok:#0F7B5F; --ok-bg:#E6F4EF; --warn:#9A5B00; --warn-bg:#FDF1DC;
}
html { scroll-behavior:smooth; }
html, body, .stApp, [class*="css"] { font-family:'Public Sans', sans-serif; color:var(--text); }
.stApp { background:var(--white); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility:hidden; height:0; }
.block-container { max-width:1200px; padding:0 1.5rem 0 1.5rem !important; }

/* ---------- Header ---------- */
.cw-util { background:var(--deep); color:#B9CAD8; font-size:.76rem; padding:.5rem 1.5rem; margin:0 -1.5rem;
  display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; }
.cw-util .st { display:inline-flex; align-items:center; gap:.4rem; color:#fff; font-weight:500; }
.cw-dot { width:7px; height:7px; border-radius:50%; background:#34D399; }
.cw-dot.off { background:#F87171; }
.st-key-topnav { background:#fff; border-bottom:1px solid var(--line); margin:0 -1.5rem; padding:.2rem 1.5rem 0 1.5rem; }
.st-key-topnav [data-testid="stHorizontalBlock"] { align-items:center; gap:.25rem; }
.cw-brand { display:flex; align-items:center; gap:1rem; padding:.7rem 0; }
.cw-bn { font-weight:800; font-size:1.55rem; color:var(--ink); letter-spacing:-.03em; line-height:1; }
.cw-bn i { font-style:normal; color:var(--blue); }
.cw-bs { font-size:.74rem; color:var(--muted); font-weight:500; border-left:1px solid var(--line);
  padding-left:1rem; line-height:1.25; }
.st-key-topnav button { background:transparent !important; border:none !important; box-shadow:none !important;
  border-radius:0 !important; padding:1.05rem .4rem !important; border-bottom:3px solid transparent !important; min-height:0; }
.st-key-topnav button p { font-weight:600; font-size:.93rem; color:var(--ink); }
.st-key-topnav button:hover p { color:var(--blue); }
.st-key-topnav button[data-testid="stBaseButton-primary"] { border-bottom:3px solid var(--blue) !important; }

/* ---------- Buttons (pill, single action colour) ---------- */
.st-key-cta_start button, .st-key-cta_end button { background:var(--blue) !important; border:none !important;
  border-radius:999px !important; padding:.9rem 2rem !important; box-shadow:none !important; }
.st-key-cta_start button p, .st-key-cta_end button p { color:#fff; font-weight:700; font-size:1rem; }
.st-key-cta_start button:hover { background:var(--blue-d) !important; }
.st-key-cta_end button { background:#fff !important; }
.st-key-cta_end button p { color:var(--deep); }
.st-key-cta_end button:hover { background:var(--surface) !important; }

/* ---------- Hero ---------- */
.st-key-hero { background:var(--surface); margin:0 -1.5rem; padding:5rem 1.5rem 5rem 1.5rem; }
.st-key-hero [data-testid="stHorizontalBlock"] { max-width:1200px; margin:0 auto; align-items:center; gap:4rem; }
.cw-h1 { font-weight:800; color:var(--ink); letter-spacing:-.035em; font-size:clamp(2.6rem, 5.4vw, 4.6rem);
  line-height:1.02; margin:0 0 1.4rem 0; }
.cw-hero-p { color:var(--text); font-size:1.15rem; line-height:1.7; max-width:540px; margin:0 0 2rem 0; }
.cw-glance { background:#fff; border-radius:8px; border-top:4px solid var(--bronze); padding:1.8rem 1.9rem;
  box-shadow:0 1px 2px rgba(18,32,46,.06), 0 12px 32px rgba(18,32,46,.07); }
.cw-glance h4 { margin:0 0 1rem 0; color:var(--ink); font-size:1.1rem; font-weight:700; }
.cw-grow { display:flex; justify-content:space-between; gap:1rem; padding:.85rem 0; border-top:1px solid var(--line); font-size:.95rem; }
.cw-grow span { color:var(--muted); }
.cw-grow b { color:var(--ink); text-align:right; }

/* ---------- Stat strip ---------- */
.cw-stats { display:grid; grid-template-columns:repeat(4,1fr); border-bottom:1px solid var(--line); }
.cw-stat { padding:2.4rem 1.6rem 2rem 0; border-right:1px solid var(--line); }
.cw-stat + .cw-stat { padding-left:1.8rem; }
.cw-stat:last-child { border-right:none; }
.cw-stat b { display:block; font-size:3rem; font-weight:800; color:var(--ink); letter-spacing:-.04em; line-height:1; }
.cw-stat span { display:block; color:var(--muted); font-size:.9rem; margin-top:.6rem; font-weight:500; }
.cw-statnote { color:var(--muted); font-size:.8rem; margin:.9rem 0 0 0; }

/* ---------- Sections ---------- */
.cw-h2 { font-weight:800; color:var(--ink); font-size:2.3rem; line-height:1.15; letter-spacing:-.03em;
  margin:5rem 0 .7rem 0; max-width:720px; }
.cw-lead { color:var(--text); font-size:1.05rem; line-height:1.7; max-width:680px; margin:0 0 2.2rem 0; }
.cw-steps { display:grid; grid-template-columns:repeat(3,1fr); gap:2.5rem; }
.cw-step .n { font-size:3.4rem; font-weight:800; color:var(--bronze); line-height:1; letter-spacing:-.04em; }
.cw-step h4 { color:var(--ink); font-size:1.15rem; margin:.9rem 0 .5rem 0; font-weight:700; padding-top:.9rem; border-top:1px solid var(--line); }
.cw-step p { color:var(--text); font-size:.97rem; line-height:1.65; margin:0; }
.cw-tiles { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; }
.cw-tile { background:var(--surface); border-radius:8px; padding:1.6rem; }
.cw-tile h5 { margin:0 0 .6rem 0; color:var(--ink); font-size:1.02rem; font-weight:700; }
.cw-tile p { margin:0; color:var(--text); font-size:.92rem; line-height:1.65; }
.cw-tech { display:grid; grid-template-columns:1fr 1fr; gap:4rem; align-items:start; }
.cw-spec { border-top:2px solid var(--ink); }
.cw-spec > div { display:flex; justify-content:space-between; gap:1rem; padding:1.05rem 0; border-bottom:1px solid var(--line); }
.cw-spec span { color:var(--muted); font-size:.95rem; }
.cw-spec b { color:var(--ink); text-align:right; font-weight:600; }
.cw-note { color:var(--muted); font-size:.85rem; line-height:1.65; margin-top:1rem; }
.st-key-cta_band { background:var(--deep); border-radius:8px; padding:3rem 3rem 3rem 3rem; margin-top:5rem; }
.st-key-cta_band h3 { color:#fff; font-size:2rem; font-weight:800; letter-spacing:-.03em; margin:0 0 .5rem 0; }
.st-key-cta_band p { color:#B9CAD8; margin:0 0 1.4rem 0; font-size:1.02rem; }

/* ---------- Assessment page ---------- */
.cw-page { background:var(--surface); margin:0 -1.5rem; padding:3rem 1.5rem 2.8rem 1.5rem; }
.cw-page .in { max-width:1200px; margin:0 auto; }
.cw-crumb { color:var(--muted); font-size:.82rem; margin-bottom:.9rem; font-weight:500; }
.cw-page h1 { color:var(--ink); font-weight:800; letter-spacing:-.035em; font-size:3rem; margin:0 0 .7rem 0; line-height:1.05; }
.cw-page p { color:var(--text); margin:0; max-width:640px; line-height:1.7; font-size:1.05rem; }
.cw-sec { display:flex; align-items:baseline; gap:.9rem; margin:.3rem 0 0 0; }
.cw-sec .n { color:var(--bronze); font-weight:800; font-size:1.05rem; }
.cw-sec-title { font-weight:700; color:var(--ink); font-size:1.25rem; letter-spacing:-.01em; }
.cw-sec-sub { color:var(--muted); font-size:.88rem; margin:.15rem 0 .9rem 0; }
div[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border:1px solid var(--line) !important;
  border-radius:8px !important; padding:.6rem .8rem; box-shadow:none; }
div[data-baseweb="input"], div[data-baseweb="select"] > div { border-radius:6px !important; background:#fff; }
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
  border-color:var(--blue) !important; box-shadow:0 0 0 1px var(--blue) !important; }
.stNumberInput label, .stSelectbox label { font-weight:600 !important; font-size:.85rem !important; color:var(--ink) !important; }
div[data-testid="stFormSubmitButton"] button { background:var(--blue); color:#fff; border:none; border-radius:999px;
  padding:1rem 1.5rem; font-weight:700; font-size:1.02rem; box-shadow:none; width:100%; transition:background .2s; }
div[data-testid="stFormSubmitButton"] button:hover { background:var(--blue-d); color:#fff; border:none; }
div[data-testid="stFormSubmitButton"] button p { color:#fff; font-weight:700; }
div[data-testid="stForm"] { border:none; padding:0; background:transparent; }
.cw-aside { background:var(--surface); border-radius:8px; padding:1.7rem; }
.cw-aside h4 { color:var(--ink); margin:0 0 .6rem 0; font-size:1.05rem; font-weight:700; }
.cw-aside ul { margin:0 0 1.5rem 0; padding-left:1.1rem; color:var(--text); font-size:.9rem; line-height:1.75; }
.cw-aside p { color:var(--text); font-size:.88rem; line-height:1.7; margin:0 0 1rem 0; }
.cw-aside a { color:var(--blue); font-weight:600; text-decoration:none; }

/* ---------- Outcome ---------- */
.cw-out { background:#fff; border:1px solid var(--line); border-radius:8px; padding:2.2rem 2.2rem 1.6rem 2.2rem; margin-top:1.5rem;
  box-shadow:0 12px 32px rgba(18,32,46,.07); }
.cw-pill { display:inline-block; font-size:.8rem; font-weight:700; padding:.3rem .85rem; border-radius:999px; }
.ok .cw-pill { background:var(--ok-bg); color:var(--ok); }
.no .cw-pill { background:var(--warn-bg); color:var(--warn); }
.cw-out h2 { font-weight:800; font-size:3.2rem; letter-spacing:-.04em; margin:.9rem 0 .7rem 0; line-height:1; color:var(--ink); }
.cw-out p { color:var(--text); line-height:1.7; margin:0 0 1.6rem 0; max-width:720px; }
.cw-sum { display:grid; grid-template-columns:repeat(4,1fr); border-top:2px solid var(--ink); }
.cw-sum div { padding:1rem 1rem .5rem 0; }
.cw-sum span { display:block; color:var(--muted); font-size:.78rem; font-weight:500; }
.cw-sum b { color:var(--ink); font-size:1.02rem; font-weight:700; }

/* ---------- Footer ---------- */
.cw-footer { background:var(--deep); color:#B9CAD8; margin:5rem -1.5rem 0 -1.5rem; padding:3.2rem 1.5rem 1.6rem 1.5rem; }
.cw-fin { max-width:1200px; margin:0 auto; }
.cw-fg { display:grid; grid-template-columns:1.5fr 1fr 1fr; gap:3rem; }
.cw-footer .brand { font-weight:800; font-size:1.5rem; color:#fff; letter-spacing:-.03em; }
.cw-footer .sub { font-size:.78rem; color:#8FB0C8; font-weight:500; margin:.3rem 0 .9rem 0; }
.cw-footer h5 { color:#fff; font-size:.92rem; margin:0 0 .7rem 0; font-weight:700; }
.cw-footer p { margin:.25rem 0; font-size:.9rem; line-height:1.65; }
.cw-footer a { color:#7FD3E6; text-decoration:none; font-weight:600; }
.cw-footer a:hover { text-decoration:underline; }
.cw-fb { border-top:1px solid rgba(255,255,255,.16); margin-top:2.2rem; padding-top:1.2rem; font-size:.78rem; color:#8FB0C8; line-height:1.75; }

/* ---------- Responsive ---------- */
@media (max-width:900px) {
  .cw-steps, .cw-tech, .cw-fg { grid-template-columns:1fr; gap:1.8rem; }
  .cw-tiles, .cw-sum, .cw-stats { grid-template-columns:repeat(2,1fr); }
  .cw-stat { border-right:none; padding-left:0 !important; }
  .st-key-hero { padding:3rem 1.5rem; }
  .cw-page h1 { font-size:2.2rem; }
  .cw-bs { display:none; }
}
@media (max-width:520px) { .cw-tiles, .cw-sum, .cw-stats { grid-template-columns:1fr; } .cw-h2 { font-size:1.8rem; } }
"""


def render(markup: str) -> None:
    """Render HTML safely: strip indentation so Markdown never treats it as a code block."""
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the saved model and fitted scaler. Returns (model, scaler, error_message)."""
    missing = [p.name for p in (MODEL_PATH, SCALER_PATH) if not p.exists()]
    if missing:
        return None, None, f"Missing required file(s): {', '.join(missing)}. Place them next to app.py."
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        return None, None, "The model files could not be loaded. They may be corrupted or built with an incompatible scikit-learn version."

    expected = len(FEATURE_COLUMNS)
    for name, obj in (("model", model), ("scaler", scaler)):
        n_in = getattr(obj, "n_features_in_", expected)
        if n_in != expected:
            return None, None, f"The saved {name} expects {n_in} features, but this app builds {expected}."
    return model, scaler, None


# ============================================================================
# FEATURE CONSTRUCTION + ASSESSMENT
# ============================================================================
def build_feature_frame(inputs: dict) -> pd.DataFrame:
    """Create the single-row DataFrame with the exact 27 columns in training order."""
    row = {col: 0 for col in FEATURE_COLUMNS}

    # Numeric features
    row["Applicant_Income"] = inputs["applicant_income"]
    row["Coapplicant_Income"] = inputs["coapplicant_income"]
    row["Age"] = inputs["age"]
    row["Dependents"] = inputs["dependents"]
    row["Existing_Loans"] = inputs["existing_loans"]
    row["Savings"] = inputs["savings"]
    row["Collateral_Value"] = inputs["collateral_value"]
    row["Loan_Amount"] = inputs["loan_amount"]
    row["Loan_Term"] = inputs["loan_term"]
    row["Education_Level"] = EDUCATION_MAP[inputs["education"]]

    # One-hot features (dropped baseline categories stay all-zero)
    for prefix, key in [
        ("Employment_Status", "employment_status"),
        ("Employer_Category", "employer_category"),
        ("Marital_Status", "marital_status"),
        ("Loan_Purpose", "loan_purpose"),
        ("Property_Area", "property_area"),
        ("Gender", "gender"),
    ]:
        col = f"{prefix}_{inputs[key]}"
        if col in row:
            row[col] = 1

    # Engineered features
    row["DTI_Ratio_sq"] = engineer_dti_sq(inputs["dti_ratio"])
    row["Credit_score_sq"] = engineer_credit_sq(inputs["credit_score"])

    frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    if list(frame.columns) != FEATURE_COLUMNS or frame.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("Feature construction mismatch")
    return frame


def run_assessment(inputs: dict, model, scaler) -> bool:
    """Apply the saved scaler and model. Returns True when Eligible."""
    frame = build_feature_frame(inputs)
    scaled = scaler.transform(frame)  # saved, already-fitted scaler only
    label = model.predict(scaled)[0]
    return int(label) == POSITIVE_CLASS


def validate_inputs(i: dict) -> list[str]:
    """Return a list of user-facing validation problems."""
    errors = []
    if i["applicant_income"] <= 0 and i["employment_status"] != "Unemployed":
        errors.append("Applicant income must be greater than zero for an employed applicant.")
    if i["loan_amount"] <= 0:
        errors.append("Requested loan amount must be greater than zero.")
    if i["loan_term"] <= 0:
        errors.append("Loan term must be greater than zero.")
    if i["employment_status"] == "Unemployed" and i["employer_category"] != "Unemployed":
        errors.append("Employer category should be 'Unemployed' when employment status is 'Unemployed'.")
    if i["employment_status"] != "Unemployed" and i["employer_category"] == "Unemployed":
        errors.append("Choose an employer category that matches the selected employment status.")
    return errors


# ============================================================================
# NAVIGATION (two-page experience via session state)
# ============================================================================
HOME, ASSESS = "home", "assessment"


def go_to(page: str) -> None:
    st.session_state["page"] = page


def money(v: float) -> str:
    return f"{CURRENCY}{v:,.0f}"


def render_header(engine_ok: bool) -> None:
    dot = "cw-dot" if engine_ok else "cw-dot off"
    label = "Assessment engine online" if engine_ok else "Assessment engine offline"
    render(f"""
    <div class="cw-util"><span>Educational machine-learning assessment. Not an official lending service.</span>
    <span class="st"><span class="{dot}"></span>{label}</span></div>
    """)
    current = st.session_state["page"]
    with st.container(key="topnav"):
        c_brand, c_home, c_assess = st.columns([6, 1.2, 2.2])
        with c_brand:
            render("""
            <div class="cw-brand"><div class="cw-bn">Credit<i>Wise</i></div>
            <div class="cw-bs">Loan Intelligence<br>Platform</div></div>
            """)
        c_home.button("Home", key="nav_home", on_click=go_to, args=(HOME,),
                      type="primary" if current == HOME else "secondary", use_container_width=True)
        c_assess.button("Eligibility Assessment", key="nav_assess", on_click=go_to, args=(ASSESS,),
                        type="primary" if current == ASSESS else "secondary", use_container_width=True)


def render_footer() -> None:
    render(f"""
    <div class="cw-footer"><div class="cw-fin">
    <div class="cw-fg">
    <div><div class="brand">CreditWise</div><div class="sub">Loan Intelligence Platform</div>
    <p>Data-driven loan eligibility assessment, built for learning and demonstration.</p></div>
    <div><h5>Developed by Hamza Malik</h5><p>Contact Developer:</p>
    <p><a href="mailto:{DEVELOPER_EMAIL}">{DEVELOPER_EMAIL}</a></p></div>
    <div><h5>Technology</h5><p>Python | Scikit-learn | Streamlit</p></div>
    </div>
    <div class="cw-fb">© 2026 CreditWise. All rights reserved.<br>
    CreditWise provides educational, machine-learning-based assessments and is not an official lending service.
    Assessment outcomes are not lending decisions or guarantees of approval.</div>
    </div></div>
    """)


# ----------------------------------------------------------------------------
# PAGE 1 - HOME
# ----------------------------------------------------------------------------
def page_home() -> None:
    stats = "".join(f'<div class="cw-stat"><b>{v}</b><span>{k}</span></div>' for k, v in EVAL_METRICS)

    with st.container(key="hero"):
        left, right = st.columns([1.2, 1])
        with left:
            render("""
            <h1 class="cw-h1">Make informed financial decisions.</h1>
            <p class="cw-hero-p">Experience a smarter approach to loan eligibility assessment. CreditWise uses
            machine learning to evaluate applicant information and provide data-driven assessment outcomes.</p>
            """)
            st.button("Start your assessment", key="cta_start", on_click=go_to, args=(ASSESS,))
        with right:
            render("""
            <div class="cw-glance"><h4>Assessment at a glance</h4>
            <div class="cw-grow"><span>Application</span><b>4 sections, 18 inputs</b></div>
            <div class="cw-grow"><span>Method</span><b>Logistic Regression</b></div>
            <div class="cw-grow"><span>Outcome</span><b>Eligible or Not Eligible</b></div>
            <div class="cw-grow"><span>Status</span><b>Educational assessment</b></div></div>
            """)

    render(f"""
    <div class="cw-stats">{stats}</div>
    <p class="cw-statnote">Model evaluation metrics measured on a held-out test set. They are not a guarantee of future performance.</p>

    <div class="cw-h2">How the assessment works</div>
    <p class="cw-lead">Three stages take an application from raw details to a clear outcome.</p>
    <div class="cw-steps">
    <div class="cw-step"><div class="n">1</div><h4>Applicant Information</h4>
    <p>Provide personal, employment and loan requirement details through a short, guided application.</p></div>
    <div class="cw-step"><div class="n">2</div><h4>Financial Evaluation</h4>
    <p>The financial profile is prepared and standardised exactly as it was when the model was trained.</p></div>
    <div class="cw-step"><div class="n">3</div><h4>Assessment Outcome</h4>
    <p>Receive an Eligible or Not Eligible status together with a summary of the submitted details.</p></div>
    </div>

    <div class="cw-h2">What the assessment considers</div>
    <p class="cw-lead">Four groups of information make up each applicant's financial profile.</p>
    <div class="cw-tiles">
    <div class="cw-tile"><h5>Personal Information</h5><p>Age, gender, marital status, dependents and education level.</p></div>
    <div class="cw-tile"><h5>Employment &amp; Income</h5><p>Employment status, employer category, applicant and co-applicant income.</p></div>
    <div class="cw-tile"><h5>Financial Profile</h5><p>Existing loans, savings, collateral, requested amount, term, credit score and debt-to-income ratio.</p></div>
    <div class="cw-tile"><h5>Loan Requirements</h5><p>The purpose of the loan and the area in which the property is located.</p></div>
    </div>

    <div class="cw-h2">The technology behind CreditWise</div>
    <div class="cw-tech"><div>
    <p class="cw-lead" style="margin-bottom:0">CreditWise is built on a Logistic Regression classifier. Applicant inputs are converted
    into the same feature set used in training and scaled with the saved StandardScaler before the model
    returns a binary outcome.</p>
    <p class="cw-note">The assessment is educational. It does not replace the judgement of a lender, and it
    does not explain why a particular outcome was reached.</p></div>
    <div class="cw-spec">
    <div><span>Algorithm</span><b>Logistic Regression</b></div>
    <div><span>Model type</span><b>Supervised Machine Learning</b></div>
    <div><span>Task</span><b>Binary Classification</b></div>
    <div><span>Framework</span><b>Scikit-learn</b></div>
    </div></div>
    """)
    with st.container(key="cta_band"):
        render("""
        <h3>Ready to begin?</h3>
        <p>The application takes only a few minutes to complete.</p>
        """)
        st.button("Go to Eligibility Assessment", key="cta_end", on_click=go_to, args=(ASSESS,))


# ----------------------------------------------------------------------------
# PAGE 2 - ASSESSMENT
# ----------------------------------------------------------------------------
def section_header(num: str, title: str, sub: str) -> None:
    render(f'<div class="cw-sec"><span class="n">{num}</span><span class="cw-sec-title">{title}</span></div>'
           f'<div class="cw-sec-sub">{sub}</div>')


def render_outcome(eligible: bool, i: dict) -> None:
    if eligible:
        css, title = "ok", "Eligible"
        text = ("Based on the submitted details, the CreditWise model classifies this profile as Eligible. "
                "This is an educational, machine-learning-based assessment, not an official lending decision "
                "or a guarantee of loan approval.")
    else:
        css, title = "no", "Not Eligible"
        text = ("Based on the submitted details, the CreditWise model classifies this profile as Not Eligible. "
                "This is an educational, machine-learning-based assessment, not an official lending decision.")
    items = [
        ("Total income", money(i["applicant_income"] + i["coapplicant_income"])),
        ("Requested amount", money(i["loan_amount"])),
        ("Loan term", f'{i["loan_term"]} months'),
        ("Credit score", str(i["credit_score"])),
        ("Debt-to-income", f'{i["dti_ratio"]:.0%}'),
        ("Savings", money(i["savings"])),
        ("Loan purpose", i["loan_purpose"]),
        ("Property area", i["property_area"]),
    ]
    cells = "".join(f"<div><span>{k}</span><b>{v}</b></div>" for k, v in items)
    render(f"""
    <div class="cw-out {css}"><span class="cw-pill">Assessment Outcome</span>
    <h2>{title}</h2><p>{text}</p>
    <div class="cw-sum">{cells}</div></div>
    """)


def render_form(model, scaler, engine_error) -> None:
    with st.form("assessment_form", clear_on_submit=False):
        with st.container(border=True):
            section_header("01", "Personal Information", "Basic details about the applicant.")
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("Applicant Age", 18, 80, 30, 1, help="Age in years (18 to 80).")
            gender = c2.selectbox("Gender", CATEGORY_OPTIONS["Gender"])
            marital = c3.selectbox("Marital Status", CATEGORY_OPTIONS["Marital_Status"])
            c4, c5 = st.columns(2)
            dependents = c4.number_input("Number of Dependents", 0, 10, 0, 1)
            education = c5.selectbox("Education Level", list(EDUCATION_MAP.keys()))

        with st.container(border=True):
            section_header("02", "Employment & Income", "Current occupation and income.")
            c1, c2 = st.columns(2)
            emp_status = c1.selectbox("Employment Status", CATEGORY_OPTIONS["Employment_Status"])
            emp_cat = c2.selectbox("Employer Category", CATEGORY_OPTIONS["Employer_Category"])
            c3, c4 = st.columns(2)
            app_income = c3.number_input(f"Applicant Income ({CURRENCY})", 0.0, 10_000_000.0, 5000.0, 500.0, format="%.0f")
            co_income = c4.number_input(f"Co-applicant Income ({CURRENCY})", 0.0, 10_000_000.0, 0.0, 500.0, format="%.0f")

        with st.container(border=True):
            section_header("03", "Financial Profile", "Existing obligations, assets and the requested loan.")
            c1, c2, c3 = st.columns(3)
            existing = c1.number_input("Existing Loans", 0, 20, 0, 1, help="Number of active loans.")
            savings = c2.number_input(f"Savings ({CURRENCY})", 0.0, 100_000_000.0, 10000.0, 1000.0, format="%.0f")
            collateral = c3.number_input(f"Collateral Value ({CURRENCY})", 0.0, 100_000_000.0, 0.0, 1000.0, format="%.0f")
            c4, c5 = st.columns(2)
            loan_amt = c4.number_input(f"Requested Loan Amount ({CURRENCY})", 0.0, 100_000_000.0, 100000.0, 5000.0, format="%.0f")
            loan_term = c5.number_input("Loan Term (months)", 1, 480, 60, 6)
            c6, c7 = st.columns(2)
            credit = c6.number_input("Credit Score", 300, 900, 700, 1, help="Typical range 300 to 900.")
            dti = c7.number_input("Debt-to-Income Ratio", 0.0, 1.0, 0.30, 0.01, format="%.2f",
                                  help="Monthly debt payments divided by monthly income, as a fraction (0.30 = 30%).")

        with st.container(border=True):
            section_header("04", "Loan Requirements", "What the loan is for and where the property is located.")
            c1, c2 = st.columns(2)
            purpose = c1.selectbox("Loan Purpose", CATEGORY_OPTIONS["Loan_Purpose"])
            area = c2.selectbox("Property Area", CATEGORY_OPTIONS["Property_Area"])

        submitted = st.form_submit_button("Submit for Eligibility Assessment", use_container_width=True)

    if not submitted:
        return
    if engine_error or model is None or scaler is None:
        st.error(engine_error or "The assessment engine is unavailable.")
        return

    inputs = {
        "age": age, "gender": gender, "marital_status": marital, "dependents": dependents,
        "education": education, "employment_status": emp_status, "employer_category": emp_cat,
        "applicant_income": app_income, "coapplicant_income": co_income, "existing_loans": existing,
        "savings": savings, "collateral_value": collateral, "loan_amount": loan_amt,
        "loan_term": loan_term, "credit_score": credit, "dti_ratio": dti,
        "loan_purpose": purpose, "property_area": area,
    }
    problems = validate_inputs(inputs)
    if problems:
        for p in problems:
            st.error(p)
        return
    try:
        eligible = run_assessment(inputs, model, scaler)
    except Exception:
        st.error("We could not complete the assessment. Please review your inputs and try again.")
        return
    render_outcome(eligible, inputs)


def page_assessment(model, scaler, engine_error) -> None:
    render("""
    <div class="cw-page"><div class="in">
    <div class="cw-crumb">Home &nbsp;/&nbsp; Eligibility Assessment</div>
    <h1>Loan Eligibility Assessment</h1>
    <p>Complete the four sections below to receive an assessment outcome. All fields are required.</p>
    </div></div>
    <div style="height:1.8rem"></div>
    """)
    main_col, side_col = st.columns([2.2, 1], gap="large")
    with main_col:
        render_form(model, scaler, engine_error)
    with side_col:
        render(f"""
        <div class="cw-aside">
        <h4>Before you begin</h4>
        <ul><li>Use the applicant's most recent figures.</li>
        <li>Debt-to-income ratio is a fraction, so 0.30 means 30%.</li>
        <li>Loan term is entered in months.</li>
        <li>Nothing you enter is stored.</li></ul>
        <h4>About this assessment</h4>
        <p>The outcome comes from a Logistic Regression model trained on historical loan data. It is educational
        and is not an official lending decision or a guarantee of approval.</p>
        <p>Questions? <a href="mailto:{DEVELOPER_EMAIL}">{DEVELOPER_EMAIL}</a></p>
        </div>
        """)


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    st.session_state.setdefault("page", HOME)
    inject_css()
    model, scaler, engine_error = load_artifacts()

    render_header(engine_error is None)
    if st.session_state["page"] == ASSESS:
        page_assessment(model, scaler, engine_error)
    else:
        page_home()
    render_footer()


if __name__ == "__main__":
    main()