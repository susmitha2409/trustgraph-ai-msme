"""
================================================================
TrustGraph AI — Dynamic Financial DNA Engine for MSMEs
ml_engine.py

Purpose:
    Trains ONE Random Forest model on the synthetic MSME dataset
    to predict a business's Financial Health Score, from which
    Risk Level, Loan Eligibility, and Recommended Loan Amount are
    deterministically derived.

    The model is kept ENTIRELY IN MEMORY (no pickle/joblib files
    written to disk) — consistent with the "in-memory or CSV only"
    architecture rule.
================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


class MLEngine:
    """
    Wraps a single Random Forest Regressor that predicts a
    Financial Health Score (0-100) for an MSME business.

    Risk Level, Loan Eligibility, and Recommended Loan Amount are
    all derived from this one score + business fundamentals, so
    the project stays true to "ONE Random Forest model."

    Usage:
        engine = MLEngine()
        metrics = engine.train(df)
        result = engine.predict(business_row_dict)
    """

    # Features used to train / predict the Financial Health Score
    FEATURE_COLUMNS = [
        "Business_Age",
        "Revenue",
        "Expenses",
        "UPI_Transactions",
        "Employee_Count",
        "Supplier_Count",
        "Customer_Count",
        "Digital_Payments_Pct",
        "Cash_Withdrawals",
        "Cashflow_Ratio",
        "Profit_Margin",
        "Business_Stability_Score",
        "Digital_Adoption_Score",
        "Customer_Stability",
        "Supplier_Diversity",
        "Growth_Score",
        "Sector_Encoded",
        "GST_Encoded",
        "Previous_Loan_Encoded",
        "Repayment_Encoded",
    ]

    # Categorical columns that need label encoding
    CATEGORICAL_COLUMNS = {
        "Sector": "Sector_Encoded",
        "GST_Status": "GST_Encoded",
        "Previous_Loan": "Previous_Loan_Encoded",
        "Repayment_Behaviour": "Repayment_Encoded",
    }

    def __init__(self, random_seed: int = 42):
        """
        Initialize the ML engine.

        Args:
            random_seed (int): Seed for reproducible training results.
        """
        self.random_seed = random_seed
        self.model = None
        self.encoders = {}
        self.is_trained = False
        self.feature_importance_ = None
        self.training_metrics_ = {}

    # ------------------------------------------------------------------
    # STEP 1: SYNTHETIC TARGET GENERATION
    # ------------------------------------------------------------------
    def _generate_target_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates the ground-truth 'Financial_Health_Score' target used
        for training, as a weighted composite of engineered features
        plus small random noise to simulate real-world unpredictability.

        Args:
            df (pd.DataFrame): Feature-engineered MSME dataset.

        Returns:
            pd.Series: Financial Health Score (0-100) per business.
        """
        repayment_penalty = df["Repayment_Behaviour"].map({
            "Excellent": 10, "Good": 5, "Average": 0,
            "Poor": -15, "Defaulted": -30, "No History": -3
        }).fillna(0)

        score = (
            (df["Cashflow_Ratio"].clip(-1, 1) * 100) * 0.20
            + df["Business_Stability_Score"] * 0.20
            + df["Digital_Adoption_Score"] * 0.15
            + df["Customer_Stability"] * 0.10
            + df["Supplier_Diversity"] * 0.10
            + df["Growth_Score"] * 0.20
            + repayment_penalty
        )

        # Add small random noise to simulate real-world variability
        noise = np.random.normal(loc=0, scale=3, size=len(df))
        score = score + noise

        return score.clip(0, 100).round(2)

    # ------------------------------------------------------------------
    # STEP 2: ENCODING CATEGORICAL FEATURES
    # ------------------------------------------------------------------
    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encodes categorical columns into numeric form using LabelEncoder.

        Args:
            df (pd.DataFrame): Dataset containing categorical columns.
            fit (bool): If True, fits new encoders. If False, reuses
                        existing fitted encoders (used at prediction time).

        Returns:
            pd.DataFrame: Dataset with additional *_Encoded columns.
        """
        df = df.copy()

        for raw_col, encoded_col in self.CATEGORICAL_COLUMNS.items():
            if fit:
                encoder = LabelEncoder()
                df[encoded_col] = encoder.fit_transform(df[raw_col].astype(str))
                self.encoders[raw_col] = encoder
            else:
                encoder = self.encoders[raw_col]
                # Handle unseen categories gracefully by mapping to a known class
                known_classes = set(encoder.classes_)
                safe_values = df[raw_col].astype(str).apply(
                    lambda x: x if x in known_classes else encoder.classes_[0]
                )
                df[encoded_col] = encoder.transform(safe_values)

        return df

    # ------------------------------------------------------------------
    # STEP 3: TRAINING
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame) -> dict:
        """
        Trains the Random Forest Regressor on the full synthetic dataset.

        Args:
            df (pd.DataFrame): Feature-engineered MSME dataset.

        Returns:
            dict: Training metrics (MAE, R2 Score, feature importances).
        """
        df = df.copy()

        # Generate the training target
        df["Financial_Health_Score"] = self._generate_target_score(df)

        # Encode categorical variables
        df = self._encode_categoricals(df, fit=True)

        X = df[self.FEATURE_COLUMNS]
        y = df["Financial_Health_Score"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed
        )

        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=14,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_seed,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        self.feature_importance_ = pd.Series(
            self.model.feature_importances_, index=self.FEATURE_COLUMNS
        ).sort_values(ascending=False)

        self.training_metrics_ = {
            "mae": round(mae, 3),
            "r2_score": round(r2, 3),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        self.is_trained = True
        return self.training_metrics_

    # ------------------------------------------------------------------
    # STEP 4: DERIVED BUSINESS LOGIC (Risk, Eligibility, Loan Amount)
    # ------------------------------------------------------------------
    def _derive_risk_level(self, health_score: float) -> str:
        """
        Maps a Financial Health Score to a Risk Level category.

        Args:
            health_score (float): Predicted Financial Health Score (0-100).

        Returns:
            str: One of "Low", "Moderate", "High", "Critical".
        """
        if health_score >= 75:
            return "Low"
        elif health_score >= 55:
            return "Moderate"
        elif health_score >= 35:
            return "High"
        else:
            return "Critical"

    def _derive_loan_eligibility(self, health_score: float, risk_level: str) -> str:
        """
        Determines whether the business is eligible for a loan.

        Args:
            health_score (float): Predicted Financial Health Score.
            risk_level (str): Derived risk category.

        Returns:
            str: "Eligible", "Conditionally Eligible", or "Not Eligible".
        """
        if risk_level in ("Low", "Moderate") and health_score >= 50:
            return "Eligible"
        elif risk_level == "High" and health_score >= 35:
            return "Conditionally Eligible"
        else:
            return "Not Eligible"

    def _derive_recommended_loan_amount(self, business_row: dict, health_score: float) -> float:
        """
        Estimates a recommended loan amount based on revenue capacity
        and the predicted financial health score.

        Args:
            business_row (dict): Raw business feature values.
            health_score (float): Predicted Financial Health Score.

        Returns:
            float: Recommended loan amount in INR.
        """
        revenue = float(business_row.get("Revenue", 0))
        health_multiplier = health_score / 100.0

        # Base loan capacity: a fraction of annual revenue, scaled by health
        base_capacity = revenue * 0.35 * health_multiplier

        # Cap the loan amount within realistic MSME loan bounds
        recommended = float(np.clip(base_capacity, 10000, 5000000))
        return round(recommended, 2)

    # ------------------------------------------------------------------
    # STEP 5: PREDICTION (single business)
    # ------------------------------------------------------------------
    def predict(self, business_row: dict) -> dict:
        """
        Predicts Financial Health Score, Risk Level, Loan Eligibility,
        and Recommended Loan Amount for a single business.

        Args:
            business_row (dict): Dictionary of business feature values
                                  (must contain all raw + engineered columns).

        Returns:
            dict: {
                "financial_health_score": float,
                "risk_level": str,
                "loan_eligibility": str,
                "recommended_loan_amount": float
            }
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        row_df = pd.DataFrame([business_row])
        row_df = self._encode_categoricals(row_df, fit=False)

        X = row_df[self.FEATURE_COLUMNS]
        health_score = float(self.model.predict(X)[0])
        health_score = round(float(np.clip(health_score, 0, 100)), 2)

        risk_level = self._derive_risk_level(health_score)
        loan_eligibility = self._derive_loan_eligibility(health_score, risk_level)
        recommended_loan = self._derive_recommended_loan_amount(business_row, health_score)

        return {
            "financial_health_score": health_score,
            "risk_level": risk_level,
            "loan_eligibility": loan_eligibility,
            "recommended_loan_amount": recommended_loan,
        }

    # ------------------------------------------------------------------
    # STEP 6: BATCH PREDICTION (whole dataset)
    # ------------------------------------------------------------------
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts Financial Health Score and derived outputs for an
        entire dataset at once (used for dashboard-wide analytics).

        Args:
            df (pd.DataFrame): Feature-engineered MSME dataset.

        Returns:
            pd.DataFrame: Original dataset with prediction columns appended.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        df = df.copy()
        encoded_df = self._encode_categoricals(df, fit=False)
        X = encoded_df[self.FEATURE_COLUMNS]

        scores = self.model.predict(X)
        scores = np.clip(scores, 0, 100).round(2)

        df["Financial_Health_Score"] = scores
        df["Risk_Level"] = [self._derive_risk_level(s) for s in scores]
        df["Loan_Eligibility"] = [
            self._derive_loan_eligibility(s, r)
            for s, r in zip(scores, df["Risk_Level"])
        ]
        df["Recommended_Loan_Amount"] = [
            self._derive_recommended_loan_amount(row, s)
            for row, s in zip(df.to_dict("records"), scores)
        ]

        return df

    # ------------------------------------------------------------------
    # STEP 7: FEATURE IMPORTANCE ACCESS
    # ------------------------------------------------------------------
    def get_feature_importance(self, top_n: int = 10) -> pd.Series:
        """
        Returns the top N most important features used by the model.

        Args:
            top_n (int): Number of top features to return.

        Returns:
            pd.Series: Feature importances sorted descending.
        """
        if self.feature_importance_ is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.feature_importance_.head(top_n)


# ------------------------------------------------------------------
# STANDALONE TEST (only runs if this file is executed directly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    from data_generator import MSMEDataGenerator

    gen = MSMEDataGenerator(num_businesses=2000)
    dataset = gen.generate_full_dataset("msme_data.csv")

    engine = MLEngine()
    metrics = engine.train(dataset)
    print("Training Metrics:", metrics)
    print("\nTop Features:\n", engine.get_feature_importance())

    sample = dataset.iloc[0].to_dict()
    result = engine.predict(sample)
    print("\nSample Prediction:", result)
