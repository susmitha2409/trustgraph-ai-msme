"""
================================================================
TrustGraph AI — Dynamic Financial DNA Engine for MSMEs
llm_report.py

Purpose:
    Generates a professional, human-readable financial report for
    an MSME business using the Groq LLM (via APIManager), based on
    the business's Financial DNA Card and SHAP explanation.

    The report covers: Business Summary, Strengths, Weaknesses,
    Risk, Recommended Loan, and Future Outlook.
================================================================
"""

from api_manager import APIManager


class LLMReportGenerator:
    """
    Builds prompts from a Financial DNA Card + SHAP explanation and
    uses the APIManager (Groq, 4-key failover) to generate a
    professional financial report in natural language.

    Usage:
        reporter = LLMReportGenerator()
        report_text = reporter.generate_report(card, explanation)
    """

    def __init__(self):
        """
        Initializes the report generator with its own APIManager
        instance for Groq access with automatic key rotation.
        """
        self.api_manager = APIManager()

    # ------------------------------------------------------------------
    # STEP 1: PROMPT CONSTRUCTION
    # ------------------------------------------------------------------
    def _build_prompt(self, card: dict, explanation: dict, business_row: dict) -> list:
        """
        Constructs the chat messages sent to the Groq LLM, embedding
        the Financial DNA Card and SHAP explanation as structured
        context so the model can ground its report in real numbers.

        Args:
            card (dict): Financial DNA Card (scores, risk, loan info).
            explanation (dict): SHAP explanation (top contributing factors).
            business_row (dict): Original raw business data.

        Returns:
            list: Chat messages formatted for the Groq API.
        """
        top_factors_text = "\n".join(
            f"- {f['feature']}: {f['direction']} impact ({f['impact']})"
            for f in explanation.get("top_factors", [])
        )

        system_prompt = (
            "You are a professional financial analyst at a bank writing "
            "concise, clear MSME loan assessment reports. Write in plain "
            "business English, avoid jargon, and be specific using the "
            "numbers provided. Do not invent data not given to you. "
            "Structure your response with these exact section headers: "
            "Business Summary, Strengths, Weaknesses, Risk Assessment, "
            "Recommended Loan, Future Outlook."
        )

        user_prompt = f"""
Generate a professional financial report for the following MSME business.

BUSINESS PROFILE:
- Name: {card.get('business_name')}
- Sector: {business_row.get('Sector')}
- Business Age: {business_row.get('Business_Age')} years
- Annual Revenue: INR {business_row.get('Revenue'):,.2f}
- Annual Expenses: INR {business_row.get('Expenses'):,.2f}
- GST Status: {business_row.get('GST_Status')}
- Employee Count: {business_row.get('Employee_Count')}
- Previous Loan: {business_row.get('Previous_Loan')}
- Repayment Behaviour: {business_row.get('Repayment_Behaviour')}

FINANCIAL DNA CARD:
- Cashflow Score: {card.get('cashflow_score')}/100
- Growth Score: {card.get('growth_score')}/100
- Trust Score: {card.get('trust_score')}/100
- Digital Score: {card.get('digital_score')}/100
- Fraud Score: {card.get('fraud_score')}/100 (lower is safer)
- Overall Health Score: {card.get('overall_health_score')}/100
- Confidence Score: {card.get('confidence_score')}/100
- Risk Level: {card.get('risk_level')}
- Loan Eligibility: {card.get('loan_eligibility')}
- Recommended Loan Amount: INR {card.get('recommended_loan_amount'):,.2f}

KEY MODEL FACTORS (from SHAP explainability):
{top_factors_text}

Write the report now, using the exact section headers requested.
Keep each section to 2-4 sentences. Be specific and reference the
numbers above where relevant.
"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.strip()},
        ]

    # ------------------------------------------------------------------
    # STEP 2: REPORT GENERATION
    # ------------------------------------------------------------------
    def generate_report(self, card: dict, explanation: dict, business_row: dict) -> str:
        """
        Generates the full financial report text for a business.

        Args:
            card (dict): Financial DNA Card from financial_dna.py.
            explanation (dict): SHAP explanation from financial_dna.py.
            business_row (dict): Raw business data dictionary.

        Returns:
            str: The generated financial report (markdown-formatted text).
        """
        messages = self._build_prompt(card, explanation, business_row)

        report_text = self.api_manager.generate_completion(
            messages=messages,
            temperature=0.4,
            max_tokens=1200,
        )

        return report_text

    # ------------------------------------------------------------------
    # STEP 3: FALLBACK REPORT (used only if no keys configured / offline)
    # ------------------------------------------------------------------
    def generate_fallback_report(self, card: dict, business_row: dict) -> str:
        """
        Produces a simple, deterministic, template-based report without
        calling any LLM — used as a graceful offline fallback so the
        UI always has something meaningful to display.

        Args:
            card (dict): Financial DNA Card.
            business_row (dict): Raw business data dictionary.

        Returns:
            str: A basic template-generated financial report.
        """
        return f"""
### Business Summary
{card.get('business_name')} operates in the {business_row.get('Sector')} sector
with {business_row.get('Business_Age')} years of operating history and an
annual revenue of INR {business_row.get('Revenue'):,.2f}.

### Strengths
Cashflow Score of {card.get('cashflow_score')}/100 and Growth Score of
{card.get('growth_score')}/100 indicate healthy business fundamentals.

### Weaknesses
Fraud Score of {card.get('fraud_score')}/100 and Digital Score of
{card.get('digital_score')}/100 highlight areas needing improvement.

### Risk Assessment
Overall Risk Level is classified as **{card.get('risk_level')}**, with a
Financial Health Score of {card.get('overall_health_score')}/100.

### Recommended Loan
Based on current financial health, a loan amount of
INR {card.get('recommended_loan_amount'):,.2f} is recommended
(Eligibility: {card.get('loan_eligibility')}).

### Future Outlook
Continued improvement in digital adoption and cashflow management is
expected to further strengthen this business's financial profile.

*Note: This is a simplified offline report. Connect valid Groq API keys for AI-generated insights.*
"""

    # ------------------------------------------------------------------
    # STEP 4: SMART WRAPPER (tries LLM, falls back automatically)
    # ------------------------------------------------------------------
    def generate_smart_report(self, card: dict, explanation: dict, business_row: dict) -> str:
        """
        Attempts to generate an AI-powered report via Groq. If no API
        keys are configured at all, falls back to the deterministic
        template report instead of showing an error to the user.

        Args:
            card (dict): Financial DNA Card.
            explanation (dict): SHAP explanation.
            business_row (dict): Raw business data dictionary.

        Returns:
            str: Final report text (AI-generated or fallback).
        """
        if not self.api_manager.has_valid_keys():
            return self.generate_fallback_report(card, business_row)

        return self.generate_report(card, explanation, business_row)


# ------------------------------------------------------------------
# STANDALONE TEST (only runs if this file is executed directly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from data_generator import MSMEDataGenerator
    from ml_engine import MLEngine
    from financial_dna import FinancialDNA

    gen = MSMEDataGenerator(num_businesses=1000)
    dataset = gen.generate_full_dataset("msme_data.csv")

    engine = MLEngine()
    engine.train(dataset)

    dna = FinancialDNA(engine)
    sample = dataset.iloc[0].to_dict()
    card = dna.generate_card(sample)
    explanation = dna.explain(sample)

    reporter = LLMReportGenerator()
    report = reporter.generate_smart_report(card, explanation, sample)

    print(report)
