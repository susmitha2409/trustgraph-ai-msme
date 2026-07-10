"""
================================================================
TrustGraph AI — MSME Financial Health Card
app.py

Purpose:
    The main Streamlit application. Ties together data_generator.py,
    ml_engine.py, financial_dna.py, api_manager.py, and llm_report.py
    into a single 5-page dashboard:

        1. Home
        2. Generate Dataset
        3. Financial DNA Card
        4. Loan Simulator
        5. AI Report

    All state (dataset, trained model, selected business) is kept
    in Streamlit's session_state — no database, no external files
    beyond the optional CSV export.

    NOTE: This file contains presentation/UI logic only. All scoring,
    ML training, SHAP explainability, and LLM report generation is
    delegated unchanged to data_generator.py, ml_engine.py,
    financial_dna.py, api_manager.py, and llm_report.py.
================================================================
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

from data_generator import MSMEDataGenerator
from ml_engine import MLEngine
from financial_dna import FinancialDNA
from llm_report import LLMReportGenerator

# ------------------------------------------------------------------
# APP CONFIG
# ------------------------------------------------------------------
load_dotenv()

# st.set_page_config() MUST be the very first Streamlit command executed,
# before any other st.* call (including touching st.secrets, which can
# itself render a warning if no secrets.toml is present).
st.set_page_config(
    page_title="TrustGraph AI | MSME Financial Health Card",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bridge Streamlit Cloud secrets into os.environ so api_manager.py
# (which reads keys via os.environ.get) works both locally with a
# .env file and when deployed on Streamlit Cloud with st.secrets.
try:
    for _key in ["GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4"]:
        if _key in st.secrets:
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass  # no secrets.toml locally — fine, .env / load_dotenv() already handled it

CSV_PATH = "msme_data.csv"

# ------------------------------------------------------------------
# BRAND / THEME CONSTANTS
# ------------------------------------------------------------------
NAVY = "#0B1F3A"
NAVY_LIGHT = "#123159"
GOLD = "#C89B3C"
GOLD_LIGHT = "#E4C878"
SLATE = "#4B5563"
BG = "#F4F6F9"
CARD_BG = "#FFFFFF"
GREEN = "#1E7B4D"
AMBER = "#B8860B"
ORANGE = "#C0602A"
RED = "#B3261E"

RISK_STYLE = {
    "Low":      {"color": GREEN,  "bg": "#E7F5EC", "label": "LOW RISK"},
    "Moderate": {"color": AMBER,  "bg": "#FBF3D9", "label": "MODERATE RISK"},
    "High":     {"color": ORANGE, "bg": "#FBEADF", "label": "HIGH RISK"},
    "Critical": {"color": RED,    "bg": "#FBE7E6", "label": "CRITICAL RISK"},
}


# ------------------------------------------------------------------
# GLOBAL STYLING
# ------------------------------------------------------------------
def inject_custom_css():
    """
    Injects bank-grade custom CSS for typography, layout, sidebar,
    cards, metrics, and buttons. Purely cosmetic — no functional
    behaviour is changed by this styling layer.
    """
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{
            background-color: {BG};
        }}

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {{
            background-color: {NAVY};
        }}
        section[data-testid="stSidebar"] * {{
            color: #E8ECF3 !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.15);
        }}
        section[data-testid="stSidebar"] .stRadio > label {{
            color: #E8ECF3 !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label {{
            padding: 6px 10px;
            border-radius: 6px;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            background-color: rgba(255,255,255,0.08);
        }}
        section[data-testid="stSidebar"] .tg-banner-success,
        section[data-testid="stSidebar"] .tg-banner-success * {{
            color: #12492E !important;
        }}
        section[data-testid="stSidebar"] .tg-banner-warning,
        section[data-testid="stSidebar"] .tg-banner-warning * {{
            color: #5C4700 !important;
        }}
        section[data-testid="stSidebar"] .tg-banner-info,
        section[data-testid="stSidebar"] .tg-banner-info * {{
            color: #0B3355 !important;
        }}
        section[data-testid="stSidebar"] .tg-banner-error,
        section[data-testid="stSidebar"] .tg-banner-error * {{
            color: #7A1C16 !important;
        }}

        /* ---------- Header masthead ---------- */
        .tg-masthead {{
            background: linear-gradient(90deg, {NAVY} 0%, {NAVY_LIGHT} 100%);
            padding: 28px 36px;
            border-radius: 10px;
            margin-bottom: 26px;
            border-left: 6px solid {GOLD};
        }}
        .tg-masthead h1 {{
            color: #FFFFFF;
            font-size: 30px;
            font-weight: 800;
            margin: 0 0 4px 0;
            letter-spacing: 0.3px;
        }}
        .tg-masthead p {{
            color: {GOLD_LIGHT};
            font-size: 14.5px;
            font-weight: 500;
            margin: 0;
            letter-spacing: 0.4px;
            text-transform: uppercase;
        }}
        .tg-masthead .tg-sub {{
            color: #C7D0DE;
            font-size: 13.5px;
            font-weight: 400;
            text-transform: none;
            margin-top: 8px;
        }}

        /* ---------- Section headers ---------- */
        .tg-section-title {{
            font-size: 19px;
            font-weight: 700;
            color: {NAVY};
            margin: 6px 0 14px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #E3E7EE;
        }}

        /* ---------- Chart card (gauges + radar) ---------- */
        div[data-testid="stPlotlyChart"] {{
            background-color: #E4EFFC;
            border: 1.5px solid #A9C6EC;
            border-radius: 10px;
            padding: 10px 6px 0 6px;
            margin-bottom: 12px;
            box-shadow: 0 2px 5px rgba(15, 31, 58, 0.10);
        }}

        /* ---------- Generic card ---------- */
        .tg-card {{
            background-color: {CARD_BG};
            border: 1px solid #E3E7EE;
            border-radius: 10px;
            padding: 20px 22px;
            box-shadow: 0 1px 3px rgba(15, 31, 58, 0.06);
            height: 100%;
        }}
        .tg-card h4 {{
            color: {NAVY};
            font-size: 15.5px;
            font-weight: 700;
            margin: 0 0 8px 0;
        }}
        .tg-card p, .tg-card li {{
            color: {SLATE};
            font-size: 13.8px;
            line-height: 1.55;
        }}

        /* ---------- Pipeline step badges ---------- */
        .tg-step {{
            display: flex;
            align-items: flex-start;
            gap: 14px;
            background-color: {CARD_BG};
            border: 1px solid #E3E7EE;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 10px;
        }}
        .tg-step .tg-num {{
            flex-shrink: 0;
            width: 30px; height: 30px;
            border-radius: 50%;
            background-color: {NAVY};
            color: {GOLD_LIGHT};
            font-weight: 700;
            font-size: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .tg-step .tg-step-text b {{
            color: {NAVY};
        }}
        .tg-step .tg-step-text {{
            color: {SLATE};
            font-size: 13.8px;
            padding-top: 4px;
        }}

        /* ---------- Status pill ---------- */
        .tg-pill {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.3px;
        }}

        /* ---------- Risk badge (large) ---------- */
        .tg-risk-badge {{
            display: inline-block;
            padding: 6px 18px;
            border-radius: 6px;
            font-weight: 800;
            font-size: 14px;
            letter-spacing: 0.6px;
        }}

        /* ---------- Custom metric card (used instead of st.metric) ---------- */
        .tg-metric-card {{
            background-color: #E4EFFC;
            border: 1.5px solid #A9C6EC;
            border-left: 4px solid {GOLD};
            border-radius: 8px;
            padding: 14px 18px 16px 18px;
            box-shadow: 0 2px 5px rgba(15, 31, 58, 0.10);
            height: 100%;
            margin-bottom: 12px;
        }}
        .tg-metric-label {{
            color: {SLATE} !important;
            font-weight: 700 !important;
            font-size: 12px !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin: 0 0 6px 0;
            opacity: 1 !important;
        }}
        .tg-metric-value {{
            color: {NAVY} !important;
            font-weight: 800 !important;
            font-size: 25px !important;
            line-height: 1.2;
            margin: 0;
            word-break: break-word;
        }}
        .tg-metric-sub {{
            color: #8B96A8 !important;
            font-size: 11.5px !important;
            margin-top: 4px;
        }}

        /* ---------- Buttons ---------- */
        .stButton > button {{
            background-color: {NAVY};
            color: #FFFFFF;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            padding: 0.5rem 1.1rem;
        }}
        .stButton > button:hover {{
            background-color: {NAVY_LIGHT};
            color: {GOLD_LIGHT};
        }}
        .stDownloadButton > button {{
            background-color: #FFFFFF;
            color: {NAVY};
            border: 1.5px solid {NAVY};
            border-radius: 6px;
            font-weight: 600;
        }}
        .stDownloadButton > button:hover {{
            background-color: {NAVY};
            color: #FFFFFF;
        }}

        /* ---------- Custom banners (theme-proof replacement for st.info/warning/success/error) ---------- */
        .tg-banner {{
            border-radius: 8px;
            padding: 12px 18px;
            margin: 10px 0;
            font-size: 14px;
            font-weight: 500;
            border-left: 5px solid;
        }}
        .tg-banner-info {{
            background-color: #E8F0FB !important;
            border-color: #2E6E8E !important;
            color: #0B3355 !important;
        }}
        .tg-banner-success {{
            background-color: #E7F5EC !important;
            border-color: {GREEN} !important;
            color: #12492E !important;
        }}
        .tg-banner-warning {{
            background-color: #FBF3D9 !important;
            border-color: {AMBER} !important;
            color: #5C4700 !important;
        }}
        .tg-banner-error {{
            background-color: #FBE7E6 !important;
            border-color: {RED} !important;
            color: #7A1C16 !important;
        }}
        .tg-banner b {{
            color: inherit !important;
        }}

        /* ---------- Force legible labels on native widgets regardless of viewer theme ---------- */
        div[data-testid="stWidgetLabel"] p,
        div[data-testid="stWidgetLabel"] label,
        .stSelectbox label, .stSlider label, .stNumberInput label,
        div[data-baseweb="select"] + div, label {{
            color: {NAVY} !important;
            font-weight: 600 !important;
            opacity: 1 !important;
        }}
        div[data-baseweb="select"] > div {{
            background-color: #FFFFFF !important;
            border-color: #CBD3E1 !important;
        }}
        div[data-baseweb="select"] * {{
            color: {NAVY} !important;
        }}
        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"] {{
            color: {SLATE} !important;
        }}
        .stSlider div[data-testid="stThumbValue"] {{
            color: {NAVY} !important;
            background-color: #FFFFFF !important;
            border: 1px solid {GOLD} !important;
        }}
        div[data-testid="stNumberInput"] input {{
            color: {NAVY} !important;
            background-color: #FFFFFF !important;
        }}
        div[data-testid="stDataFrame"] {{
            background-color: #FFFFFF !important;
        }}
        div[data-testid="stCaptionContainer"], .stCaption, small {{
            color: {SLATE} !important;
            opacity: 1 !important;
        }}
        .tg-report-letterhead {{
            background-color: {CARD_BG};
            border: 1px solid #E3E7EE;
            border-top: 5px solid {NAVY};
            border-radius: 8px;
            padding: 10px 26px 2px 26px;
            margin-bottom: 4px;
        }}
        .tg-report-letterhead h3 {{
            color: {NAVY};
            margin-bottom: 2px;
        }}
        .tg-report-letterhead .tg-tag {{
            color: {SLATE};
            font-size: 12.5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .tg-report-body {{
            background-color: {CARD_BG};
            border: 1px solid #E3E7EE;
            border-radius: 0 0 8px 8px;
            padding: 6px 26px 24px 26px;
            margin-top: -8px;
        }}
        .tg-report-body h1, .tg-report-body h2, .tg-report-body h3,
        .tg-report-body h4, .tg-report-body h5 {{
            color: {NAVY} !important;
            font-weight: 700 !important;
            margin-top: 18px;
            margin-bottom: 6px;
        }}
        .tg-report-body p, .tg-report-body li, .tg-report-body span {{
            color: {SLATE} !important;
            font-size: 14.5px !important;
            line-height: 1.6;
        }}
        .tg-report-body strong, .tg-report-body b {{
            color: {NAVY} !important;
        }}

        /* ---------- Footer ---------- */
        .tg-footer {{
            text-align: center;
            color: #9AA5B1;
            font-size: 12px;
            padding: 18px 0 6px 0;
        }}
    </style>
    """, unsafe_allow_html=True)


def render_masthead(title: str, tagline: str, sub: str = ""):
    """Renders the navy/gold bank-style masthead header."""
    sub_html = f'<div class="tg-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="tg-masthead">
        <h1>🏦 {title}</h1>
        <p>{tagline}</p>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_section_title(text: str):
    st.markdown(f'<div class="tg-section-title">{text}</div>', unsafe_allow_html=True)


def render_risk_badge(risk_level: str) -> str:
    style = RISK_STYLE.get(risk_level, {"color": SLATE, "bg": "#EEE", "label": risk_level})
    return (
        f'<span class="tg-risk-badge" style="background-color:{style["bg"]};'
        f'color:{style["color"]};">{style["label"]}</span>'
    )


def render_metric(container, label: str, value, sub: str = ""):
    """
    Renders a professional, clearly-labelled score/amount card in the
    given Streamlit container (a column or the main page). Used instead
    of st.metric so the heading above every figure is always legible.

    Args:
        container: A Streamlit column/container object (or `st` itself).
        label (str): The heading describing what the value represents,
                     e.g. "Overall Health Score", "Recommended Loan".
        value: The figure to display, e.g. "31.0/100", "₹196,865".
        sub (str): Optional small helper caption shown under the value.
    """
    sub_html = f'<div class="tg-metric-sub">{sub}</div>' if sub else ""
    container.markdown(f"""
    <div class="tg-metric-card">
        <div class="tg-metric-label">{label}</div>
        <div class="tg-metric-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_banner(kind: str, text: str, container=None):
    """
    Renders a theme-proof status banner (replaces st.info/warning/success/
    error, whose text color is controlled by the viewer's light/dark
    Streamlit theme and can become illegible against the pastel background).

    Args:
        kind (str): One of "info", "success", "warning", "error".
        text (str): Message to display (HTML-safe plain text/markdown-lite).
        container: Optional Streamlit container to render into (defaults to st).
    """
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}
    icon = icons.get(kind, "ℹ️")
    target = container if container is not None else st
    target.markdown(
        f'<div class="tg-banner tg-banner-{kind}">{icon}&nbsp; {text}</div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------------------
def init_session_state():
    """
    Initializes all required keys in Streamlit's session_state so
    the app never crashes on first load or page navigation.
    """
    defaults = {
        "dataset": None,
        "ml_engine": None,
        "financial_dna": None,
        "training_metrics": None,
        "selected_business_index": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()
inject_custom_css()


# ------------------------------------------------------------------
# CACHED / REUSABLE HELPERS  (unchanged backend logic / wiring)
# ------------------------------------------------------------------
def train_pipeline(df: pd.DataFrame):
    """
    Trains the ML engine and builds the FinancialDNA wrapper, storing
    both in session_state for reuse across pages.

    Args:
        df (pd.DataFrame): Feature-engineered MSME dataset.
    """
    engine = MLEngine()
    metrics = engine.train(df)
    dna = FinancialDNA(engine)

    st.session_state["ml_engine"] = engine
    st.session_state["financial_dna"] = dna
    st.session_state["training_metrics"] = metrics


def make_gauge(title: str, value: float, color: str = NAVY) -> go.Figure:
    """
    Builds a Plotly gauge chart for a 0-100 score.

    Args:
        title (str): Gauge title.
        value (float): Score value (0-100).
        color (str): Bar color for the gauge.

    Returns:
        go.Figure: Plotly gauge figure.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 15, "color": SLATE, "family": "Inter"}},
        number={"font": {"size": 30, "color": NAVY, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#C7D0DE"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#F4F6F9",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#FBEAEA"},
                {"range": [40, 70], "color": "#FBF3D9"},
                {"range": [70, 100], "color": "#E7F5EC"},
            ],
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def make_radar_chart(card: dict) -> go.Figure:
    """
    Builds a radar/spider chart summarizing all Financial DNA sub-scores.

    Args:
        card (dict): Financial DNA Card.

    Returns:
        go.Figure: Plotly radar chart figure.
    """
    categories = ["Cashflow", "Growth", "Trust", "Digital", "Confidence"]
    values = [
        card["cashflow_score"],
        card["growth_score"],
        card["trust_score"],
        card["digital_score"],
        card["confidence_score"],
    ]
    # Close the loop for radar chart
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        line_color=NAVY,
        fillcolor="rgba(11, 31, 58, 0.18)",
        marker=dict(color=GOLD, size=6),
    ))
    fig.update_layout(
        polar={
            "bgcolor": "#FFFFFF",
            "radialaxis": {"visible": True, "range": [0, 100], "gridcolor": "#E3E7EE"},
            "angularaxis": {"gridcolor": "#E3E7EE"},
        },
        showlegend=False,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": SLATE},
    )
    return fig


def calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> float:
    """
    Calculates the EMI (Equated Monthly Installment) for a loan.

    Args:
        principal (float): Loan amount.
        annual_rate (float): Annual interest rate (percentage).
        tenure_months (int): Loan tenure in months.

    Returns:
        float: Monthly EMI amount.
    """
    if tenure_months <= 0:
        return 0.0
    monthly_rate = (annual_rate / 100) / 12
    if monthly_rate == 0:
        return round(principal / tenure_months, 2)
    emi = principal * monthly_rate * (1 + monthly_rate) ** tenure_months
    emi = emi / (((1 + monthly_rate) ** tenure_months) - 1)
    return round(emi, 2)


def calculate_approval_probability(health_score: float, requested_amount: float,
                                    recommended_amount: float) -> float:
    """
    Estimates a simple approval probability (%) based on how the
    requested loan amount compares to the model's recommended amount
    and the business's financial health score.

    Args:
        health_score (float): Financial Health Score (0-100).
        requested_amount (float): Loan amount requested by user in simulator.
        recommended_amount (float): Model's recommended loan amount.

    Returns:
        float: Approval probability percentage (0-100).
    """
    if recommended_amount <= 0:
        ratio_penalty = 50
    else:
        excess_ratio = requested_amount / recommended_amount
        if excess_ratio <= 1:
            ratio_penalty = 0
        else:
            ratio_penalty = min((excess_ratio - 1) * 60, 60)

    probability = health_score - ratio_penalty
    return round(max(min(probability, 99), 1), 2)


# ------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------
st.sidebar.markdown(
    f"""
    <div style="padding: 6px 0 2px 0;">
        <div style="font-size: 21px; font-weight: 800; color: #FFFFFF;">🏦 TrustGraph AI</div>
        <div style="font-size: 12px; color: {GOLD_LIGHT}; letter-spacing: 0.4px; margin-top: 2px;">
            MSME FINANCIAL HEALTH CARD
        </div>
        <div style="font-size: 11px; color: #9FB0C9; margin-top: 6px;">
            IDBI Innovate · AI/ML Credit Assessment
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Generate Dataset", "Financial DNA Card", "Loan Simulator", "AI Report"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown('<div style="font-size:12px; color:#9FB0C9; text-transform:uppercase; letter-spacing:0.4px; margin-bottom:6px;">System Status</div>', unsafe_allow_html=True)

if st.session_state["dataset"] is not None:
    render_banner("success", f"Dataset loaded: {len(st.session_state['dataset']):,} businesses", container=st.sidebar)
else:
    render_banner("warning", "No dataset loaded yet", container=st.sidebar)

if st.session_state["ml_engine"] is not None and st.session_state["ml_engine"].is_trained:
    render_banner("success", "Model trained", container=st.sidebar)
else:
    render_banner("warning", "Model not trained yet", container=st.sidebar)

st.sidebar.markdown(
    """
    <div style="font-size: 11px; color: #7E8BA3; margin-top: 18px; line-height:1.5;">
        Built for the IDBI Innovate Hackathon — MSME credit evaluation using
        alternate data (GST · UPI · AA · EPFO) in place of traditional financials.
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# PAGE 1: HOME
# ------------------------------------------------------------------
if page == "Home":
    render_masthead(
        "TrustGraph AI",
        "AI/ML-Driven MSME Financial Health Card",
        "Built for New-to-Credit (NTC) &amp; New-to-Bank (NTB) enterprises using alternate data — "
        "designed for ULI · OCEN · Account Aggregator ecosystem integration.",
    )

    render_section_title("The Problem")
    st.markdown(f"""
    <div class="tg-card">
    <p>
    Traditional MSME credit evaluation depends on formal financial documents that many
    New-to-Credit and New-to-Bank enterprises simply don't have — or maintain poorly.
    Despite rich alternate data being available (GST, UPI, Account Aggregator, EPFO, etc.),
    the absence of a <b>unified assessment framework</b> results in high rejection rates,
    missed viable borrowers, weak portfolio diversification, and slower financial inclusion.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    render_section_title("How TrustGraph AI Addresses This")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="tg-card">
            <h4>📊 Alternate Data Layer</h4>
            <p>Aggregates GST registration, UPI transaction behaviour, digital payment
            adoption, supplier/customer footprint, and repayment history — in place of
            traditional bank statements and ITRs.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="tg-card">
            <h4>🌲 Multidimensional Scoring</h4>
            <p>A single Random Forest model produces a Financial Health Score, from which
            Cashflow, Growth, Trust, Digital Adoption, and Fraud Risk sub-scores and a
            Recommended Loan Amount are derived.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="tg-card">
            <h4>🔍 Explainable Decisions</h4>
            <p>SHAP-based factor attribution shows credit officers exactly why a score
            was produced, and a Groq-powered LLM narrates it as a plain-English report
            for underwriting review.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    render_section_title("Alignment with ULI / OCEN / Account Aggregator Ecosystem")
    st.markdown(f"""
    <div class="tg-card">
    <p>
    The Financial DNA Card is designed to sit downstream of standardised alternate-data
    feeds — Account Aggregator consent flows, GST/UPI signals, and OCEN-based loan
    origination — so it can plug into the Unified Lending Interface (ULI) pipeline for
    near real-time credit assessment, rather than requiring a bespoke data integration
    per lender.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    render_section_title("Assessment Pipeline")

    steps = [
        ("1", "Aggregate", "Synthetic MSME data simulating GST, UPI, employee, supplier, and repayment signals is ingested (Generate Dataset)."),
        ("2", "Score", "A single Random Forest model predicts a 0–100 Financial Health Score and derives Risk Level, Eligibility, and Recommended Loan Amount."),
        ("3", "Explain", "SHAP explainability breaks the score down into the top contributing factors, in business-friendly language (Financial DNA Card)."),
        ("4", "Simulate", "Credit officers can test EMI, tenure, and approval probability against different requested loan amounts (Loan Simulator)."),
        ("5", "Report", "A Groq-powered LLM drafts a professional underwriting narrative — Summary, Strengths, Weaknesses, Risk, Recommendation, Outlook (AI Report)."),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div class="tg-step">
            <div class="tg-num">{num}</div>
            <div class="tg-step-text"><b>{title}</b> — {desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    render_banner("info", "This build uses synthetic MSME data end-to-end for demonstration — no real business or personal information is used anywhere.")


# ------------------------------------------------------------------
# PAGE 2: GENERATE DATASET
# ------------------------------------------------------------------
elif page == "Generate Dataset":
    render_masthead("Generate Dataset", "Synthetic MSME Data & Model Training")

    render_section_title("Configure & Train")
    col1, col2 = st.columns([2, 1])
    with col1:
        num_businesses = st.slider(
            "Number of businesses to generate",
            min_value=1000, max_value=20000, value=20000, step=1000
        )
    with col2:
        st.write("")
        st.write("")
        generate_clicked = st.button("🚀 Generate & Train", type="primary", use_container_width=True)

    if generate_clicked:
        with st.spinner("Generating synthetic MSME data..."):
            generator = MSMEDataGenerator(num_businesses=num_businesses)
            df = generator.generate_full_dataset(CSV_PATH)
            st.session_state["dataset"] = df
            st.session_state["selected_business_index"] = 0

        with st.spinner("Training Random Forest model..."):
            train_pipeline(df)

        render_banner("success", f"Generated {len(df):,} businesses and trained the model successfully!")

    if st.session_state["dataset"] is not None:
        df = st.session_state["dataset"]
        st.write("")
        render_section_title("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.write("")
        render_section_title("Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        render_metric(col1, "Total Businesses", f"{len(df):,}")
        render_metric(col2, "Avg Revenue", f"₹{df['Revenue'].mean():,.0f}")
        render_metric(col3, "Avg Business Age", f"{df['Business_Age'].mean():.1f} yrs")
        render_metric(col4, "Sectors Covered", df["Sector"].nunique())

        if st.session_state["training_metrics"] is not None:
            st.write("")
            render_section_title("Model Performance")
            metrics = st.session_state["training_metrics"]
            m1, m2, m3, m4 = st.columns(4)
            render_metric(m1, "Mean Absolute Error", metrics["mae"], "Lower is better")
            render_metric(m2, "R² Score", metrics["r2_score"], "Closer to 1 is better")
            render_metric(m3, "Training Set Size", f"{metrics['train_size']:,}")
            render_metric(m4, "Test Set Size", f"{metrics['test_size']:,}")

        st.write("")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Dataset as CSV",
            data=csv_bytes,
            file_name="msme_data.csv",
            mime="text/csv",
        )
    else:
        render_banner("warning", "No dataset generated yet. Click 'Generate &amp; Train' above to begin.")


# ------------------------------------------------------------------
# PAGE 3: FINANCIAL DNA CARD
# ------------------------------------------------------------------
elif page == "Financial DNA Card":
    render_masthead("Financial DNA Card", "Multidimensional Credit Health Profile")

    if st.session_state["dataset"] is None or st.session_state["financial_dna"] is None:
        render_banner("warning", "Please generate a dataset and train the model first (see 'Generate Dataset' page).")
    else:
        df = st.session_state["dataset"]
        dna = st.session_state["financial_dna"]

        business_names = df["Business_Name"].tolist()
        selected_name = st.selectbox(
            "Select a business",
            business_names,
            index=st.session_state["selected_business_index"],
        )
        selected_index = business_names.index(selected_name)
        st.session_state["selected_business_index"] = selected_index

        business_row = df.iloc[selected_index].to_dict()

        with st.spinner("Computing Financial DNA Card..."):
            card = dna.generate_card(business_row)

        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between; margin: 6px 0 18px 0;">
            <div style="font-size:20px; font-weight:800; color:{NAVY};">{card['business_name']}</div>
            {render_risk_badge(card['risk_level'])}
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        render_metric(col1, "Overall Health Score", f"{card['overall_health_score']}/100", "Model-predicted, 0–100 scale")
        render_metric(col2, "Loan Eligibility", card["loan_eligibility"])
        render_metric(col3, "Recommended Loan Amount", f"₹{card['recommended_loan_amount']:,.0f}")

        col4, col5 = st.columns(2)
        render_metric(col4, "Confidence Score", f"{card['confidence_score']}/100", "Based on data volume & history depth")
        render_metric(col5, "Risk Level", card["risk_level"])

        st.write("")
        render_section_title("📊 Score Breakdown")

        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(make_gauge("Cashflow Score", card["cashflow_score"], NAVY), use_container_width=True)
        with g2:
            st.plotly_chart(make_gauge("Growth Score", card["growth_score"], GREEN), use_container_width=True)
        with g3:
            st.plotly_chart(make_gauge("Trust Score", card["trust_score"], GOLD), use_container_width=True)

        g4, g5, g6 = st.columns(3)
        with g4:
            st.plotly_chart(make_gauge("Digital Score", card["digital_score"], "#2E6E8E"), use_container_width=True)
        with g5:
            st.plotly_chart(make_gauge("Fraud Score (lower=safer)", card["fraud_score"], RED), use_container_width=True)
        with g6:
            st.plotly_chart(make_gauge("Overall Health", card["overall_health_score"], ORANGE), use_container_width=True)

        st.write("")
        render_section_title("🕸️ Financial DNA Radar")
        st.plotly_chart(make_radar_chart(card), use_container_width=True)

        st.write("")
        render_section_title("🔍 SHAP Explainability")
        with st.spinner("Computing SHAP explanation..."):
            explanation = dna.explain(business_row)

        st.markdown(f"""
        <div class="tg-card" style="margin-bottom:14px;">
            <p style="margin:0;"><b>Base value:</b> {explanation['base_value']} &nbsp;→&nbsp;
            <b>Predicted value:</b> {explanation['predicted_value']}</p>
        </div>
        """, unsafe_allow_html=True)

        for factor in explanation["top_factors"]:
            icon = "⬆️" if factor["direction"] == "positive" else "⬇️"
            impact_color = GREEN if factor["direction"] == "positive" else RED
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        background-color:#FFFFFF; border:1px solid #E3E7EE; border-radius:6px;
                        padding:9px 16px; margin-bottom:6px;">
                <span style="color:{NAVY}; font-weight:600; font-size:13.8px;">{icon} {factor['feature']}</span>
                <span style="color:{impact_color}; font-weight:700; font-size:13.5px;">{factor['impact']}</span>
            </div>
            """, unsafe_allow_html=True)

        # Store selected business context for AI Report page
        st.session_state["last_card"] = card
        st.session_state["last_explanation"] = explanation
        st.session_state["last_business_row"] = business_row


# ------------------------------------------------------------------
# PAGE 4: LOAN SIMULATOR
# ------------------------------------------------------------------
elif page == "Loan Simulator":
    render_masthead("Loan Simulator", "EMI & Approval Probability Modelling")

    if st.session_state["dataset"] is None or st.session_state["financial_dna"] is None:
        render_banner("warning", "Please generate a dataset and train the model first (see 'Generate Dataset' page).")
    else:
        df = st.session_state["dataset"]
        dna = st.session_state["financial_dna"]

        business_names = df["Business_Name"].tolist()
        selected_name = st.selectbox(
            "Select a business",
            business_names,
            index=st.session_state["selected_business_index"],
            key="simulator_business_select",
        )
        selected_index = business_names.index(selected_name)
        business_row = df.iloc[selected_index].to_dict()
        card = dna.generate_card(business_row)

        render_section_title("📌 Business Snapshot")
        col1, col2 = st.columns(2)
        with col1:
            render_metric(st, "Recommended Loan Amount", f"₹{card['recommended_loan_amount']:,.0f}",
                          "Maximum amount the model supports for this business")
            render_metric(st, "Financial Health Score", f"{card['overall_health_score']}/100",
                          "0–100 composite score from the ML model")
        with col2:
            render_metric(st, "Risk Level", card["risk_level"],
                          "Low / Moderate / High / Critical")
            render_metric(st, "Loan Eligibility", card["loan_eligibility"],
                          "Eligible / Conditionally Eligible / Not Eligible")

        st.write("")
        render_section_title("🎛️ Simulate a Loan")
        st.caption("Adjust the requested amount, interest rate, and tenure below to see how repayment terms and approval odds change.")

        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amount = st.number_input(
                "Requested Loan Amount (₹)", min_value=10000, max_value=10000000,
                value=int(card["recommended_loan_amount"]), step=5000,
            )
            st.caption("Amount the business is asking to borrow")
        with c2:
            interest_rate = st.slider("Annual Interest Rate (%)", 4.0, 24.0, 10.5, 0.1)
            st.caption("Yearly interest rate to be applied")
        with c3:
            tenure_months = st.slider("Repayment Tenure (months)", 3, 84, 24)
            st.caption("Total months to repay the loan in full")

        emi = calculate_emi(loan_amount, interest_rate, tenure_months)
        total_payment = round(emi * tenure_months, 2)
        total_interest = round(total_payment - loan_amount, 2)
        approval_prob = calculate_approval_probability(
            card["overall_health_score"], loan_amount, card["recommended_loan_amount"]
        )

        st.write("")
        render_section_title("📊 Simulation Results")
        r1, r2, r3, r4 = st.columns(4)
        render_metric(r1, "Monthly EMI", f"₹{emi:,.2f}", "Fixed amount payable every month")
        render_metric(r2, "Total Interest Payable", f"₹{total_interest:,.2f}", "Interest cost over the full tenure")
        render_metric(r3, "Total Repayment", f"₹{total_payment:,.2f}", "Principal + total interest")
        render_metric(r4, "Approval Probability", f"{approval_prob}%", "Model-estimated likelihood of sanction")

        if approval_prob >= 70:
            render_banner("success", f"High likelihood of approval ({approval_prob}%) — risk level: {card['risk_level']}. The requested amount is well within the model's recommended capacity.")
        elif approval_prob >= 40:
            render_banner("warning", f"Moderate likelihood of approval ({approval_prob}%) — consider reducing the requested amount closer to the ₹{card['recommended_loan_amount']:,.0f} recommendation, or reviewing tenure/rate.")
        else:
            render_banner("error", f"Low likelihood of approval ({approval_prob}%) — the requested amount exceeds what this business's financial health score currently supports (recommended: ₹{card['recommended_loan_amount']:,.0f}).")

        st.write("")
        render_section_title("🧭 How to Read This")
        st.markdown(f"""
        <div class="tg-card">
            <p style="margin:0;">
            <b>Approval Probability</b> compares the requested amount against the model's
            recommended loan capacity for this business — the closer the request is to (or
            below) the recommended amount, the higher the probability. <b>Monthly EMI</b> and
            <b>Total Interest Payable</b> update live as you change the rate or tenure above,
            so you can test different repayment structures before presenting an offer to the customer.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.plotly_chart(make_gauge("Approval Probability", approval_prob, GREEN), use_container_width=True)


# ------------------------------------------------------------------
# PAGE 5: AI REPORT
# ------------------------------------------------------------------
elif page == "AI Report":
    render_masthead("AI-Generated Financial Report", "Underwriting Narrative — Groq LLM")

    if st.session_state["dataset"] is None or st.session_state["financial_dna"] is None:
        render_banner("warning", "Please generate a dataset and train the model first (see 'Generate Dataset' page).")
    else:
        df = st.session_state["dataset"]
        dna = st.session_state["financial_dna"]

        business_names = df["Business_Name"].tolist()
        selected_name = st.selectbox(
            "Select a business",
            business_names,
            index=st.session_state["selected_business_index"],
            key="report_business_select",
        )
        selected_index = business_names.index(selected_name)
        business_row = df.iloc[selected_index].to_dict()

        keys_configured = sum(
            1 for k in ["GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4"]
            if os.environ.get(k, "").strip()
        )
        if keys_configured == 0:
            render_banner(
                "warning",
                "No Groq API keys detected in environment variables. "
                "A simplified offline report will be generated instead. "
                "Add GROQ_API_KEY_1 through GROQ_API_KEY_4 to your .env file for full AI reports."
            )
        else:
            render_banner("info", f"{keys_configured} Groq API key(s) configured with automatic failover.")

        generate_report_clicked = st.button("🧠 Generate AI Report", type="primary")

        if generate_report_clicked:
            with st.spinner("Computing Financial DNA Card..."):
                card = dna.generate_card(business_row)
                explanation = dna.explain(business_row)

            with st.spinner("Generating professional report via Groq..."):
                reporter = LLMReportGenerator()
                report_text = reporter.generate_smart_report(card, explanation, business_row)

            st.session_state["generated_report"] = report_text
            st.session_state["report_card"] = card

        if "generated_report" in st.session_state:
            st.write("")
            card = st.session_state.get("report_card", {})
            risk_level = card.get("risk_level", "")

            st.markdown(f"""
            <div class="tg-report-letterhead">
                <h3>📋 MSME Credit Assessment Report</h3>
                <div class="tg-tag">{selected_name} &nbsp;·&nbsp; Prepared by TrustGraph AI &nbsp;·&nbsp;
                {render_risk_badge(risk_level) if risk_level else ""}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                f'<div class="tg-report-body">\n\n{st.session_state["generated_report"]}\n\n</div>',
                unsafe_allow_html=True,
            )

            st.write("")
            st.download_button(
                "⬇️ Download Report as Text",
                data=st.session_state["generated_report"],
                file_name=f"{selected_name}_financial_report.txt",
                mime="text/plain",
            )

st.markdown('<div class="tg-footer">TrustGraph AI — Prototype for IDBI Innovate Hackathon · Built entirely on synthetic MSME data</div>', unsafe_allow_html=True)
