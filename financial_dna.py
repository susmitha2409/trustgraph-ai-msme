"""
================================================================
TrustGraph AI — Dynamic Financial DNA Engine for MSMEs
financial_dna.py

Purpose:
    Builds the "Financial DNA Card" for a business — a compact set
    of interpretable sub-scores (Cashflow, Growth, Trust, Digital,
    Fraud, Overall Health, Confidence) plus Risk Level and
    Recommended Loan Amount.

    Also wraps SHAP to generate simple, human-readable explainability
    for why the model produced a given Financial Health Score.
================================================================
"""

import numpy as np
import pandas as pd
import shap


class FinancialDNA:
    """
    Computes the Financial DNA Card for a single MSME business and
    provides SHAP-based explainability on top of a trained MLEngine.

    Usage:
        dna = FinancialDNA(ml_engine)
        card = dna.generate_card(business_row)
        explanation = dna.explain(business_row)
    """

    def __init__(self, ml_engine):
        """
        Initialize the Financial DNA generator.

        Args:
            ml_engine (MLEngine): A trained MLEngine instance
                                   (must have is_trained == True).
        """
        self.ml_engine = ml_engine
        self._explainer = None  # SHAP TreeExplainer, built lazily

    # ------------------------------------------------------------------
    # STEP 1: SUB-SCORE CALCULATIONS
    # ------------------------------------------------------------------
    def _cashflow_score(self, row: dict) -> float:
        """
        Computes the Cashflow Score (0-100) from the cashflow ratio.

        Args:
            row (dict): Business feature values.

        Returns:
            float: Cashflow score.
        """
        ratio = float(row.get("Cashflow_Ratio", 0))
        score = (ratio + 1) / 2 * 100  # rescale from [-1, 1] to [0, 100]
        return round(float(np.clip(score, 0, 100)), 2)

    def _growth_score(self, row: dict) -> float:
        """
        Retrieves the Growth Score (already engineered in data_generator).

        Args:
            row (dict): Business feature values.

        Returns:
            float: Growth score.
        """
        return round(float(np.clip(row.get("Growth_Score", 0), 0, 100)), 2)

    def _trust_score(self, row: dict) -> float:
        """
        Computes a Trust Score (0-100) combining business stability,
        GST registration, and repayment history.

        Args:
            row (dict): Business feature values.

        Returns:
            float: Trust score.
        """
        stability = float(row.get("Business_Stability_Score", 0))

        repayment_map = {
            "Excellent": 100, "Good": 80, "Average": 60,
            "Poor": 30, "Defaulted": 5, "No History": 50
        }
        repayment_score = repayment_map.get(row.get("Repayment_Behaviour", "No History"), 50)

        gst_bonus = 10 if row.get("GST_Status") == "Registered" else 0

        trust = (stability * 0.4) + (repayment_score * 0.5) + gst_bonus * 0.1
        return round(float(np.clip(trust, 0, 100)), 2)

    def _digital_score(self, row: dict) -> float:
        """
        Retrieves the Digital Adoption Score (already engineered).

        Args:
            row (dict): Business feature values.

        Returns:
            float: Digital score.
        """
        return round(float(np.clip(row.get("Digital_Adoption_Score", 0), 0, 100)), 2)

    def _fraud_score(self, row: dict) -> float:
        """
        Computes a Fraud Risk Score (0-100, LOWER is safer) based on
        suspicious cash-heavy behaviour relative to digital activity.

        Note: This is a simple heuristic proxy for demo purposes only,
        NOT a real fraud detection model.

        Args:
            row (dict): Business feature values.

        Returns:
            float: Fraud risk score (0 = very low risk, 100 = very high risk).
        """
        cash = float(row.get("Cash_Withdrawals", 0))
        revenue = float(row.get("Revenue", 1)) or 1
        digital_pct = float(row.get("Digital_Payments_Pct", 50))

        cash_ratio = np.clip((cash * 12) / revenue, 0, 1)  # annualized cash vs revenue
        low_digital_penalty = (100 - digital_pct) * 0.4

        fraud_risk = (cash_ratio * 100 * 0.6) + (low_digital_penalty * 0.4)
        return round(float(np.clip(fraud_risk, 0, 100)), 2)

    def _confidence_score(self, row: dict) -> float:
        """
        Computes a Confidence Score (0-100) reflecting how much data-backed
        certainty exists behind the prediction — based on business age,
        transaction volume, and customer/supplier base size.

        Args:
            row (dict): Business feature values.

        Returns:
            float: Confidence score.
        """
        age = float(row.get("Business_Age", 0))
        upi = float(row.get("UPI_Transactions", 0))
        customers = float(row.get("Customer_Count", 0))

        age_component = np.minimum(age / 10.0, 1.0) * 100
        upi_component = np.minimum(upi / 300.0, 1.0) * 100
        customer_component = np.minimum(customers / 500.0, 1.0) * 100

        confidence = (age_component * 0.4) + (upi_component * 0.3) + (customer_component * 0.3)
        return round(float(np.clip(confidence, 0, 100)), 2)

    # ------------------------------------------------------------------
    # STEP 2: FULL DNA CARD GENERATION
    # ------------------------------------------------------------------
    def generate_card(self, row: dict) -> dict:
        """
        Generates the complete Financial DNA Card for a single business.

        Args:
            row (dict): Raw + engineered feature values for one business.

        Returns:
            dict: Financial DNA Card containing all sub-scores plus
                  the ML-derived Risk Level and Recommended Loan Amount.
        """
        prediction = self.ml_engine.predict(row)

        card = {
            "business_name": row.get("Business_Name", "Unknown"),
            "cashflow_score": self._cashflow_score(row),
            "growth_score": self._growth_score(row),
            "trust_score": self._trust_score(row),
            "digital_score": self._digital_score(row),
            "fraud_score": self._fraud_score(row),
            "overall_health_score": prediction["financial_health_score"],
            "confidence_score": self._confidence_score(row),
            "risk_level": prediction["risk_level"],
            "loan_eligibility": prediction["loan_eligibility"],
            "recommended_loan_amount": prediction["recommended_loan_amount"],
        }
        return card

    # ------------------------------------------------------------------
    # STEP 3: SHAP EXPLAINABILITY
    # ------------------------------------------------------------------
    def _build_explainer(self):
        """
        Lazily builds and caches a SHAP TreeExplainer for the trained
        Random Forest model (expensive to build, so only done once).
        """
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.ml_engine.model)

    def explain(self, row: dict, top_n: int = 5) -> dict:
        """
        Generates a simple SHAP-based explanation for a single business's
        Financial Health Score prediction.

        Args:
            row (dict): Business feature values.
            top_n (int): Number of top contributing features to return.

        Returns:
            dict: {
                "base_value": float,
                "predicted_value": float,
                "top_factors": [
                    {"feature": str, "impact": float, "direction": "positive"/"negative"},
                    ...
                ]
            }
        """
        if not self.ml_engine.is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        self._build_explainer()

        row_df = pd.DataFrame([row])
        row_df = self.ml_engine._encode_categoricals(row_df, fit=False)
        X = row_df[self.ml_engine.FEATURE_COLUMNS]

        shap_values = self._explainer.shap_values(X)
        # shap_values shape: (1, num_features) for a regressor
        values = shap_values[0]

        base_value = float(self._explainer.expected_value)
        predicted_value = float(base_value + values.sum())

        # Build a readable feature -> impact mapping
        impacts = list(zip(self.ml_engine.FEATURE_COLUMNS, values))
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)

        top_factors = []
        for feature, impact in impacts[:top_n]:
            top_factors.append({
                "feature": self._humanize_feature_name(feature),
                "impact": round(float(impact), 3),
                "direction": "positive" if impact >= 0 else "negative",
            })

        return {
            "base_value": round(base_value, 2),
            "predicted_value": round(predicted_value, 2),
            "top_factors": top_factors,
        }

    @staticmethod
    def _humanize_feature_name(feature_name: str) -> str:
        """
        Converts internal feature column names into human-readable labels.

        Args:
            feature_name (str): Raw feature column name.

        Returns:
            str: Human-readable feature name.
        """
        mapping = {
            "Business_Age": "Business Age",
            "Revenue": "Revenue",
            "Expenses": "Expenses",
            "UPI_Transactions": "UPI Transaction Volume",
            "Employee_Count": "Employee Count",
            "Supplier_Count": "Supplier Count",
            "Customer_Count": "Customer Count",
            "Digital_Payments_Pct": "Digital Payments %",
            "Cash_Withdrawals": "Cash Withdrawals",
            "Cashflow_Ratio": "Cashflow Ratio",
            "Profit_Margin": "Profit Margin",
            "Business_Stability_Score": "Business Stability",
            "Digital_Adoption_Score": "Digital Adoption",
            "Customer_Stability": "Customer Stability",
            "Supplier_Diversity": "Supplier Diversity",
            "Growth_Score": "Growth Score",
            "Sector_Encoded": "Business Sector",
            "GST_Encoded": "GST Registration Status",
            "Previous_Loan_Encoded": "Previous Loan History",
            "Repayment_Encoded": "Repayment Behaviour",
        }
        return mapping.get(feature_name, feature_name)


# ------------------------------------------------------------------
# STANDALONE TEST (only runs if this file is executed directly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    from data_generator import MSMEDataGenerator
    from ml_engine import MLEngine

    gen = MSMEDataGenerator(num_businesses=2000)
    dataset = gen.generate_full_dataset("msme_data.csv")

    engine = MLEngine()
    engine.train(dataset)

    dna = FinancialDNA(engine)
    sample = dataset.iloc[0].to_dict()

    card = dna.generate_card(sample)
    print("Financial DNA Card:")
    for k, v in card.items():
        print(f"  {k}: {v}")

    explanation = dna.explain(sample)
    print("\nSHAP Explanation:")
    print(f"  Base Value: {explanation['base_value']}")
    print(f"  Predicted Value: {explanation['predicted_value']}")
    print("  Top Factors:")
    for factor in explanation["top_factors"]:
        print(f"    {factor}")
