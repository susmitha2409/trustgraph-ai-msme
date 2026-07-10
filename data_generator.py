"""
================================================================
TrustGraph AI — Dynamic Financial DNA Engine for MSMEs
data_generator.py

Purpose:
    Generates a fully synthetic dataset of MSME (Micro, Small &
    Medium Enterprise) businesses, and performs feature engineering
    to derive financial health indicators used later by ml_engine.py
    and financial_dna.py.

    NO real data is used anywhere in this project. Everything is
    generated using controlled random distributions to simulate
    realistic MSME behaviour.
================================================================
"""

import numpy as np
import pandas as pd
import os


class MSMEDataGenerator:
    """
    Generates synthetic MSME business data and engineers financial
    features on top of it.

    Usage:
        generator = MSMEDataGenerator(num_businesses=20000)
        df = generator.generate_dataset()
        df = generator.engineer_features(df)
        generator.save_to_csv(df, "msme_data.csv")
    """

    # Sectors an MSME can belong to (kept realistic & diverse)
    SECTORS = [
        "Retail", "Manufacturing", "Textiles", "Food & Beverage",
        "Electronics", "Agriculture", "Construction", "IT Services",
        "Handicrafts", "Logistics", "Healthcare", "Education"
    ]

    # Repayment behaviour categories
    REPAYMENT_BEHAVIOURS = ["Excellent", "Good", "Average", "Poor", "Defaulted"]

    def __init__(self, num_businesses: int = 20000, random_seed: int = 42):
        """
        Initialize the data generator.

        Args:
            num_businesses (int): Number of synthetic MSME records to generate.
            random_seed (int): Seed for reproducibility of random data.
        """
        self.num_businesses = num_businesses
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    # ------------------------------------------------------------------
    # STEP 1: RAW SYNTHETIC DATA GENERATION
    # ------------------------------------------------------------------
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generates the raw synthetic MSME dataset with all base columns.

        Returns:
            pd.DataFrame: Raw synthetic MSME dataset.
        """
        n = self.num_businesses

        # Business Name -> simple synthetic naming pattern
        business_names = [f"MSME-{str(i).zfill(6)}" for i in range(1, n + 1)]

        # Business Age (in years) - most businesses are young to mid-aged
        business_age = np.random.gamma(shape=2.5, scale=3.0, size=n).round(1)
        business_age = np.clip(business_age, 0.5, 40)

        # Sector - randomly assigned
        sector = np.random.choice(self.SECTORS, size=n)

        # Revenue (annual, in INR) - log-normal to simulate realistic skew
        revenue = np.random.lognormal(mean=13.5, sigma=0.9, size=n)
        revenue = np.clip(revenue, 50000, 50000000)

        # Expenses - correlated with revenue but with variability
        expense_ratio = np.random.normal(loc=0.75, scale=0.12, size=n)
        expense_ratio = np.clip(expense_ratio, 0.35, 1.15)
        expenses = revenue * expense_ratio

        # GST Status - most active MSMEs are GST registered
        gst_status = np.random.choice(
            ["Registered", "Not Registered"], size=n, p=[0.72, 0.28]
        )

        # UPI Transactions (monthly count) - digital adoption proxy
        upi_transactions = np.random.poisson(lam=180, size=n)

        # Employee Count - most MSMEs are small
        employee_count = np.random.gamma(shape=1.5, scale=4.0, size=n).astype(int)
        employee_count = np.clip(employee_count, 1, 250)

        # Supplier Count
        supplier_count = np.random.poisson(lam=8, size=n)
        supplier_count = np.clip(supplier_count, 1, 60)

        # Customer Count
        customer_count = np.random.poisson(lam=150, size=n)
        customer_count = np.clip(customer_count, 1, 5000)

        # Digital Payments (% of total transactions that are digital)
        digital_payments_pct = np.random.beta(a=5, b=3, size=n) * 100

        # Cash Withdrawals (monthly, in INR)
        cash_withdrawals = np.random.lognormal(mean=9.5, sigma=1.0, size=n)
        cash_withdrawals = np.clip(cash_withdrawals, 0, 2000000)

        # Previous Loan (Yes/No)
        previous_loan = np.random.choice(["Yes", "No"], size=n, p=[0.55, 0.45])

        # Repayment Behaviour - only meaningful if previous_loan == Yes
        repayment_behaviour = np.random.choice(
            self.REPAYMENT_BEHAVIOURS, size=n, p=[0.25, 0.30, 0.25, 0.13, 0.07]
        )
        # If no previous loan, mark as "No History"
        repayment_behaviour = np.where(
            previous_loan == "No", "No History", repayment_behaviour
        )

        # Latitude / Longitude - roughly bounded within Indian geography
        latitude = np.random.uniform(8.0, 35.0, size=n).round(6)
        longitude = np.random.uniform(68.0, 97.0, size=n).round(6)

        df = pd.DataFrame({
            "Business_Name": business_names,
            "Business_Age": business_age,
            "Sector": sector,
            "Revenue": revenue.round(2),
            "Expenses": expenses.round(2),
            "GST_Status": gst_status,
            "UPI_Transactions": upi_transactions,
            "Employee_Count": employee_count,
            "Supplier_Count": supplier_count,
            "Customer_Count": customer_count,
            "Digital_Payments_Pct": digital_payments_pct.round(2),
            "Cash_Withdrawals": cash_withdrawals.round(2),
            "Previous_Loan": previous_loan,
            "Repayment_Behaviour": repayment_behaviour,
            "Latitude": latitude,
            "Longitude": longitude,
        })

        return df

    # ------------------------------------------------------------------
    # STEP 2: FEATURE ENGINEERING
    # ------------------------------------------------------------------
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives financial health indicator features from raw MSME data.

        Features created:
            - Cashflow_Ratio
            - Profit_Margin
            - Business_Stability_Score
            - Digital_Adoption_Score
            - Customer_Stability
            - Supplier_Diversity
            - Growth_Score

        Args:
            df (pd.DataFrame): Raw synthetic MSME dataset.

        Returns:
            pd.DataFrame: Dataset enriched with engineered features.
        """
        df = df.copy()

        # --- Cashflow Ratio: Revenue vs Expenses efficiency ---
        df["Cashflow_Ratio"] = (
            (df["Revenue"] - df["Expenses"]) / df["Revenue"]
        ).clip(-1, 1)

        # --- Profit Margin (%) ---
        df["Profit_Margin"] = (
            ((df["Revenue"] - df["Expenses"]) / df["Revenue"]) * 100
        ).clip(-100, 100)

        # --- Business Stability Score (0-100) ---
        # Based on business age (older = more stable, with diminishing returns)
        age_score = np.minimum(df["Business_Age"] / 20.0, 1.0) * 100
        gst_bonus = np.where(df["GST_Status"] == "Registered", 10, 0)
        df["Business_Stability_Score"] = np.clip(
            (age_score * 0.8) + gst_bonus, 0, 100
        ).round(2)

        # --- Digital Adoption Score (0-100) ---
        # Combines digital payment %, UPI transaction volume
        upi_normalized = np.minimum(df["UPI_Transactions"] / 400.0, 1.0) * 100
        df["Digital_Adoption_Score"] = np.clip(
            (df["Digital_Payments_Pct"] * 0.6) + (upi_normalized * 0.4), 0, 100
        ).round(2)

        # --- Customer Stability (0-100) ---
        # More customers + more suppliers relative to business size = stable base
        customer_normalized = np.minimum(df["Customer_Count"] / 1000.0, 1.0) * 100
        df["Customer_Stability"] = customer_normalized.round(2)

        # --- Supplier Diversity (0-100) ---
        supplier_normalized = np.minimum(df["Supplier_Count"] / 25.0, 1.0) * 100
        df["Supplier_Diversity"] = supplier_normalized.round(2)

        # --- Growth Score (0-100) ---
        # Composite of revenue scale, employee growth proxy, and profit margin
        revenue_normalized = np.minimum(df["Revenue"] / 10000000.0, 1.0) * 100
        employee_normalized = np.minimum(df["Employee_Count"] / 100.0, 1.0) * 100
        profit_component = np.clip(df["Profit_Margin"], 0, 100)

        df["Growth_Score"] = np.clip(
            (revenue_normalized * 0.4)
            + (employee_normalized * 0.3)
            + (profit_component * 0.3),
            0, 100
        ).round(2)

        return df

    # ------------------------------------------------------------------
    # STEP 3: SAVE TO CSV
    # ------------------------------------------------------------------
    def save_to_csv(self, df: pd.DataFrame, filepath: str = "msme_data.csv") -> str:
        """
        Saves the dataset to a CSV file for persistence between sessions.

        Args:
            df (pd.DataFrame): Dataset to save.
            filepath (str): Destination CSV path.

        Returns:
            str: The filepath where the data was saved.
        """
        df.to_csv(filepath, index=False)
        return filepath

    # ------------------------------------------------------------------
    # STEP 4: LOAD FROM CSV (if already generated)
    # ------------------------------------------------------------------
    def load_from_csv(self, filepath: str = "msme_data.csv") -> pd.DataFrame:
        """
        Loads a previously saved synthetic dataset from CSV.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded dataset, or None if file doesn't exist.
        """
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        return None

    # ------------------------------------------------------------------
    # STEP 5: FULL PIPELINE (convenience method)
    # ------------------------------------------------------------------
    def generate_full_dataset(self, filepath: str = "msme_data.csv") -> pd.DataFrame:
        """
        Runs the full pipeline: generate raw data -> engineer features -> save.

        Args:
            filepath (str): Where to save the resulting CSV.

        Returns:
            pd.DataFrame: Final feature-engineered dataset.
        """
        df = self.generate_dataset()
        df = self.engineer_features(df)
        self.save_to_csv(df, filepath)
        return df


# ------------------------------------------------------------------
# STANDALONE TEST (only runs if this file is executed directly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    generator = MSMEDataGenerator(num_businesses=20000)
    dataset = generator.generate_full_dataset("msme_data.csv")
    print(f"Generated {len(dataset)} synthetic MSME records.")
    print(dataset.head())
    print("\nColumns:", list(dataset.columns))
