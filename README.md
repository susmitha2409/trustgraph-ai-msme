# TrustGraph AI — Dynamic Financial DNA Engine for MSMEs

**A synthetic-data ML + LLM platform that scores MSME (Micro, Small & Medium Enterprise) financial health, explains its decisions, and generates bank-ready credit reports — built for the IDBI Innovate Hackathon.**

🔗 **Live App:** [Add your Streamlit Cloud URL here after deployment]

---

## 📌 Overview

TrustGraph AI simulates a real-world MSME lending intelligence system entirely on **synthetic data** (no real financial data is used anywhere). It generates 20,000 synthetic businesses, trains a Random Forest model to predict a **Financial Health Score**, explains every prediction with **SHAP**, and produces a natural-language **credit assessment report** using the **Groq LLM API** — all wrapped in a 5-page Streamlit dashboard.

## ✨ Features

- **Synthetic MSME Dataset Generator** — 20,000 realistic businesses across 12 sectors with revenue, expenses, digital adoption, repayment history, and geography
- **Financial DNA Card** — 7 interpretable sub-scores: Cashflow, Growth, Trust, Digital Adoption, Fraud Risk, Overall Health, and Confidence
- **One Random Forest Model** — predicts Financial Health Score (0–100), from which Risk Level, Loan Eligibility, and Recommended Loan Amount are deterministically derived
- **SHAP Explainability** — every prediction comes with the top contributing factors, in plain English
- **Loan Simulator** — live EMI, total interest, and approval-probability modelling
- **AI-Generated Credit Reports** — Groq LLM produces a full underwriting narrative (Business Summary, Strengths, Weaknesses, Risk, Recommended Loan, Outlook), with a deterministic offline fallback if no API key is available
- **4-Key Groq API Failover** — automatic round-robin rotation across 4 API keys with retry logic, so a single rate-limited key never breaks the app

## 🏗️ Architecture

```
┌─────────────────────┐
│  data_generator.py   │  Synthetic MSME dataset (20,000 rows) + feature engineering
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│    ml_engine.py       │  Random Forest Regressor → Financial Health Score,
│                       │  Risk Level, Loan Eligibility, Recommended Loan Amount
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  financial_dna.py     │  Financial DNA Card (7 sub-scores) + SHAP explainability
└──────────┬───────────┘
           │
┌──────────▼───────────┐      ┌───────────────────────┐
│   llm_report.py       │─────▶│    api_manager.py      │  4-key Groq round-robin
│  (prompt construction)│      │  (rotation + failover) │  + retry + graceful fallback
└──────────┬───────────┘      └───────────────────────┘
           │
┌──────────▼───────────┐
│       app.py           │  Streamlit UI — 5 pages:
│                        │  Home · Generate Dataset · Financial DNA Card ·
│                        │  Loan Simulator · AI Report
└───────────────────────┘
```

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend / Dashboard | Streamlit |
| Data Generation | NumPy, Pandas |
| Machine Learning | scikit-learn (Random Forest Regressor) |
| Explainability | SHAP |
| Visualization | Plotly, Matplotlib |
| LLM Reports | Groq API (`llama-3.3-70b-versatile`) |
| Config | python-dotenv |

## 📂 Project Structure

```
idbi/
├── app.py                 # Streamlit UI — all 5 pages
├── data_generator.py      # Synthetic MSME dataset + feature engineering
├── ml_engine.py            # Random Forest training + prediction
├── financial_dna.py        # Financial DNA Card + SHAP explainability
├── llm_report.py            # Groq prompt construction + report generation
├── api_manager.py           # 4-key Groq round-robin failover manager
├── requirements.txt          # Python dependencies
└── .env                       # GROQ_API_KEY_1..4 (not committed)
```

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/susmitha2409/trustgraph-ai-msme.git
cd trustgraph-ai-msme
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API keys
Create a `.env` file in the project root:
```env
GROQ_API_KEY_1=your_key_here
GROQ_API_KEY_2=your_key_here
GROQ_API_KEY_3=your_key_here
GROQ_API_KEY_4=your_key_here
```
> The app works without any keys too — the AI Report page falls back to a deterministic offline report if no keys are configured.

### 5. Run the app
```bash
streamlit run app.py
```

## ☁️ Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (`.env` is gitignored — never commit real keys)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select this repo → main file `app.py`
3. Add your 4 Groq keys under **Advanced settings → Secrets** in TOML format
4. Deploy

## 📊 Model Details

- **Algorithm:** Random Forest Regressor (`n_estimators=200`, `max_depth=14`)
- **Target:** Synthetic Financial Health Score (0–100), a weighted composite of engineered features with added noise for realism
- **Features:** 20 engineered features spanning cashflow, stability, digital adoption, growth, and categorical business attributes
- **Explainability:** SHAP TreeExplainer, top-5 contributing factors surfaced per prediction

## ⚠️ Disclaimer

This project uses **100% synthetic data** generated programmatically. No real MSME, financial, or personal data is used anywhere in this repository. It is a hackathon prototype, not a production credit-scoring system.

## 📄 License

[Add your chosen license here — e.g. MIT]

## 🙏 Acknowledgements

Built for the **IDBI Innovate Hackathon**.
