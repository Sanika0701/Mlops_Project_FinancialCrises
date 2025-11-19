import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")

def load_csv(name):
    path = RAW_DIR / name
    if not path.exists():
        print(f"❌ Missing file: {name}")
        return None
    return pd.read_csv(path, parse_dates=["Date"] if "Date" in open(path).read() else None)


def basic_info(df, name):
    print("\n" + "="*70)
    print(f"🔍 DIAGNOSIS: {name}")
    print("="*70)

    print(f"Rows:     {len(df):,}")
    print(f"Columns:  {len(df.columns)}")
    print(f"Date range: {df['Date'].min()} → {df['Date'].max()}")
    print("")

    # Missing values
    missing = df.isna().mean().sort_values(ascending=False)
    print("Missing values (%):")
    print(missing[missing > 0].round(3))

    # Duplicate timestamps
    if df["Date"].duplicated().any():
        print("\n⚠️ Duplicate dates found:")
        print(df[df["Date"].duplicated()].head())


def company_coverage(df):
    print("\n" + "="*70)
    print("🏢 COMPANY COVERAGE (PRICE DATA)")
    print("="*70)

    gp = df.groupby("Company")["Date"]
    coverage = gp.agg(["min", "max", "count"])
    coverage["years"] = (coverage["max"] - coverage["min"]).dt.days / 365

    print(coverage.sort_values("min").head(20))
    print("\nCompanies with shortest history:")
    print(coverage.sort_values("count").head(15))

    return coverage


def fundamentals_coverage(df, label):
    print("\n" + "="*70)
    print(f"📊 FUNDAMENTALS COVERAGE — {label}")
    print("="*70)

    gp = df.groupby("Company")["Date"]
    summary = gp.agg(["min", "max", "count"])
    summary["years"] = (summary["max"] - summary["min"]).dt.days / 365

    print(summary.sort_values("count").head(15))

    return summary


def main():

    print("\n======= RAW DATA DIAGNOSIS TOOL =======")

    fred = load_csv("fred_raw.csv")
    market = load_csv("market_raw.csv")
    prices = load_csv("company_prices_raw.csv")
    income = load_csv("company_income_raw.csv")
    balance = load_csv("company_balance_raw.csv")

    if fred is not None:
        basic_info(fred, "FRED Macro Data")

    if market is not None:
        basic_info(market, "Market Data (VIX/SP500)")

    if prices is not None:
        basic_info(prices, "Company Prices (Quarterly)")
        coverage_prices = company_coverage(prices)

    if income is not None:
        basic_info(income, "Income Statement Data")
        inc_cov = fundamentals_coverage(income, "Income")

    if balance is not None:
        basic_info(balance, "Balance Sheet Data")
        bal_cov = fundamentals_coverage(balance, "Balance Sheet")

    df = pd.read_csv("data/processed/features_engineered.csv")

    cols = ["TED_Spread_mean", "TED_Spread_max", "TED_Spread_std"]

    missing_summary = (
        df[cols]
        .isna()
        .mean()           # fraction of missing
        .round(4) * 100   # convert to %
    )

    print(missing_summary)


    print("\n🎉 Diagnosis complete! Review the printed summaries.\n")
        # import pandas as pd

    # Load your file (change path if needed)
    df = pd.read_csv("data/processed/features_engineered.csv")

    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter rows between 2023 and 2025
    df_2325 = df[(df['Date'].dt.year >= 2023) & (df['Date'].dt.year <= 2025)]

    # Columns to check
    cols = ["TED_Spread_mean", "TED_Spread_max", "TED_Spread_std"]

    # Missing percentage
    missing_summary = df_2325[cols].isna().mean() * 100

    print(missing_summary)


if __name__ == "__main__":
    main()
