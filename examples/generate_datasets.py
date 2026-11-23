"""Generate the realistic synthetic demo datasets shipped in ``examples/data``.

Run:  python examples/generate_datasets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data"


def customer_churn(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    tenure = np.clip(rng.gamma(2.0, 12, n), 1, 72).round().astype(int)
    internet = rng.choice(["fiber", "dsl", "none"], n, p=[0.45, 0.4, 0.15])
    monthly = np.round(20 + (internet == "fiber") * 45 + (internet == "dsl") * 25 + rng.normal(0, 12, n).clip(-15, 40), 2)
    tickets = rng.poisson(1.2 + (internet == "fiber") * 0.8, n)
    last_login = np.clip(rng.exponential(9, n), 0, 120).round().astype(int)
    session = np.clip(rng.normal(32, 14, n), 1, None).round(1)
    products = rng.integers(1, 5, n)
    payment = rng.choice(["card", "bank_transfer", "upi", "electronic_check"], n, p=[0.3, 0.2, 0.3, 0.2])
    age = rng.integers(18, 80, n)
    logit = (
        -1.2
        + 1.6 * (contract == "month-to-month")
        - 1.1 * (contract == "two-year")
        - 0.045 * tenure
        + 0.018 * (monthly - 60)
        + 0.38 * tickets
        + 0.035 * last_login
        - 0.025 * (session - 30)
        - 0.25 * (products - 2)
        + 0.5 * (payment == "electronic_check")
        + rng.normal(0, 0.6, n)
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    start = pd.Timestamp("2024-01-01")
    df = pd.DataFrame({
        "customer_id": [f"CUST-{100000 + i}" for i in range(n)],
        "signup_date": (start + pd.to_timedelta(rng.integers(0, 900, n), unit="D")).strftime("%Y-%m-%d"),
        "age": age,
        "gender": rng.choice(["female", "male"], n),
        "contract": contract,
        "tenure_months": tenure,
        "internet_service": internet,
        "monthly_charges": monthly,
        "total_charges": np.round(monthly * tenure * rng.uniform(0.95, 1.05, n), 2),
        "payment_method": payment,
        "num_products": products,
        "support_tickets": tickets,
        "avg_session_minutes": session,
        "days_since_last_login": last_login,
        "churned": np.where(churn, "yes", "no"),
    })
    # realistic messiness
    df.loc[rng.random(n) < 0.03, "total_charges"] = np.nan
    df.loc[rng.random(n) < 0.02, "avg_session_minutes"] = np.nan
    return df


def house_prices(n: int = 2000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    city = rng.choice(["Delhi", "Mumbai", "Bengaluru", "Pune", "Chandigarh", "Indore"], n,
                      p=[0.22, 0.2, 0.22, 0.14, 0.1, 0.12])
    city_mult = pd.Series(city).map({"Mumbai": 2.1, "Delhi": 1.6, "Bengaluru": 1.5, "Pune": 1.15,
                                     "Chandigarh": 1.0, "Indore": 0.75}).to_numpy()
    area = np.clip(rng.lognormal(7.0, 0.35, n), 350, 4500).round()
    bedrooms = np.clip((area / 550 + rng.normal(0, 0.7, n)).round(), 1, 6).astype(int)
    age = rng.integers(0, 35, n)
    floor = rng.integers(0, 25, n)
    metro_km = np.round(rng.exponential(3.5, n), 2)
    furnishing = rng.choice(["unfurnished", "semi", "full"], n, p=[0.35, 0.45, 0.2])
    parking = rng.integers(0, 3, n)
    price = (
        area * 6500 * city_mult
        * (1 - 0.008 * age)
        * (1 + 0.06 * (furnishing == "full") + 0.025 * (furnishing == "semi"))
        * (1 - 0.025 * np.minimum(metro_km, 10))
        * (1 + 0.04 * parking + 0.004 * floor)
        * rng.lognormal(0, 0.12, n)
    )
    return pd.DataFrame({
        "listing_id": np.arange(1, n + 1),
        "city": city,
        "area_sqft": area,
        "bedrooms": bedrooms,
        "property_age_years": age,
        "floor": floor,
        "distance_to_metro_km": metro_km,
        "furnishing": furnishing,
        "parking_spots": parking,
        "price_inr": price.round(-3),
    })


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    customer_churn().to_csv(OUT / "customer_churn.csv", index=False)
    house_prices().to_csv(OUT / "house_prices.csv", index=False)
    print(f"Wrote datasets to {OUT}")
