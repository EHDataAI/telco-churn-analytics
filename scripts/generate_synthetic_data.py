"""Generate the public synthetic TELCO churn dataset.

The dataset is inspired by aggregate patterns of IBM's fictional Telco Customer Churn sample,
but contains no copied source rows. It uses a prediction-time data contract and a temporal split.
"""

import argparse
import csv
import math
import random
from calendar import monthrange
from datetime import date
from pathlib import Path

SEED = 20260924
SIZES = {"train": 4225, "validation": 1409, "test": 1409}
TARGET_RATES = {"train": 0.2653, "validation": 0.2730, "test": 0.2810}

COLUMNS = [
    "Customer ID",
    "Snapshot Date",
    "Age",
    "Gender",
    "Married",
    "Number of Dependents",
    "Number of Referrals",
    "Tenure in Months",
    "Contract",
    "Payment Method",
    "Paperless Billing",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Internet Type",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Unlimited Data",
    "Avg Monthly GB Download",
    "Avg Monthly Long Distance Charges",
    "Offer",
    "Preperiod Satisfaction Score",
    "Monthly Charge",
    "Total Charges",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Total Refunds",
    "Total Revenue",
    "Churn Next 30D",
]


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def weighted_choice(rng, items):
    u = rng.random() * sum(w for _, w in items)
    acc = 0.0
    for value, weight in items:
        acc += weight
        if u <= acc:
            return value
    return items[-1][0]


def month_end(year, month):
    return date(year, month, monthrange(year, month)[1])


def split_dates(split):
    months = {"train": [1, 2, 3, 4, 5, 6], "validation": [7, 8], "test": [9, 10]}[split]
    return [month_end(2025, m) for m in months]


def generate_predictors(rng, snapshot, customer_number):
    while True:
        age = round(rng.gauss(46.5, 16.7))
        if 18 <= age <= 80:
            break
    gender = "Male" if rng.random() < 0.505 else "Female"

    if age < 30:
        p_married = 0.25
    elif age < 45:
        p_married = 0.50
    elif age < 65:
        p_married = 0.58
    else:
        p_married = 0.45
    married = int(rng.random() < p_married)

    if 23 <= age <= 58 and rng.random() < (0.50 if married else 0.22):
        dependents = min(7, 1 + int(rng.expovariate(0.95)))
    else:
        dependents = 0

    tenure = int(clamp(1 + round(rng.betavariate(0.80, 0.95) * 71), 1, 72))

    p_any_referral = clamp(0.28 + 0.22 * married + 0.004 * tenure, 0.20, 0.72)
    referrals = min(11, 1 + int(rng.expovariate(0.28))) if rng.random() < p_any_referral else 0

    if tenure < 12:
        contract = weighted_choice(
            rng, [("Month-to-Month", 0.78), ("One Year", 0.18), ("Two Year", 0.04)]
        )
    elif tenure < 36:
        contract = weighted_choice(
            rng, [("Month-to-Month", 0.60), ("One Year", 0.26), ("Two Year", 0.14)]
        )
    else:
        contract = weighted_choice(
            rng, [("Month-to-Month", 0.31), ("One Year", 0.23), ("Two Year", 0.46)]
        )

    payment = weighted_choice(
        rng, [("Bank Withdrawal", 0.555), ("Credit Card", 0.390), ("Mailed Check", 0.055)]
    )
    paperless = int(rng.random() < 0.592)

    phone = int(rng.random() < 0.903)
    internet = int(rng.random() < 0.783)
    if not phone and not internet:
        internet = 1

    multiple = int(phone and rng.random() < clamp(0.30 + 0.005 * tenure, 0.30, 0.66))
    month_index = snapshot.month

    if internet:
        fiber_share = 0.54 + 0.004 * (month_index - 1)
        internet_type = weighted_choice(
            rng, [("Fiber Optic", fiber_share), ("DSL", 0.30), ("Cable", 1.0 - fiber_share - 0.30)]
        )
        online_security = int(rng.random() < 0.366)
        online_backup = int(rng.random() < 0.440)
        device_protection = int(rng.random() < 0.439)
        premium_support = int(rng.random() < 0.370)
        streaming_tv = int(rng.random() < 0.491)
        streaming_movies = int(rng.random() < 0.495)
        streaming_music = int(rng.random() < 0.451)
        unlimited = int(rng.random() < 0.860)

        gb_base = {"DSL": 16, "Cable": 22, "Fiber Optic": 30}[internet_type]
        avg_gb = int(
            round(
                clamp(
                    gb_base
                    + 7 * (streaming_tv + streaming_movies + streaming_music)
                    + 7 * unlimited
                    - 0.12 * max(age - 50, 0)
                    + rng.gauss(0, 8),
                    1,
                    85,
                )
            )
        )
    else:
        internet_type = "None"
        online_security = online_backup = device_protection = premium_support = 0
        streaming_tv = streaming_movies = streaming_music = unlimited = 0
        avg_gb = 0

    avg_long_distance = round(rng.uniform(1.00, 49.99), 2) if phone else 0.0

    if tenure < 12:
        offer = weighted_choice(
            rng,
            [
                ("None", 0.40),
                ("Offer A", 0.18),
                ("Offer B", 0.18),
                ("Offer C", 0.08),
                ("Offer D", 0.06),
                ("Offer E", 0.10),
            ],
        )
    elif tenure < 36:
        offer = weighted_choice(
            rng,
            [
                ("None", 0.52),
                ("Offer A", 0.08),
                ("Offer B", 0.15),
                ("Offer C", 0.08),
                ("Offer D", 0.08),
                ("Offer E", 0.09),
            ],
        )
    else:
        offer = weighted_choice(
            rng,
            [
                ("None", 0.64),
                ("Offer A", 0.04),
                ("Offer B", 0.07),
                ("Offer C", 0.05),
                ("Offer D", 0.10),
                ("Offer E", 0.10),
            ],
        )

    charge = 0.0
    if phone:
        charge += 18.5 + 10.0 * multiple
    if internet:
        charge += {"DSL": 25.0, "Cable": 35.0, "Fiber Optic": 50.0}[internet_type]
        charge += (
            5.5 * online_security
            + 5.0 * online_backup
            + 5.0 * device_protection
            + 6.0 * premium_support
        )
        charge += 6.5 * (streaming_tv + streaming_movies + streaming_music) + 8.0 * unlimited

    charge *= 1.0 + 0.002 * (month_index - 1)
    charge = round(clamp(charge + rng.gauss(0, 3.5), 18.0, 125.0), 2)

    satisfaction_latent = 3.45 + 0.20 * premium_support + 0.10 * online_security
    satisfaction_latent -= 0.18 * int(charge > 90) + 0.18 * int(internet_type == "Fiber Optic")
    satisfaction_latent += 0.10 * int(contract == "Two Year") + rng.gauss(0, 0.95)
    satisfaction = int(clamp(round(satisfaction_latent), 1, 5))

    billing_factor = clamp(rng.gauss(0.99, 0.035), 0.88, 1.08)
    total_charges = round(charge * tenure * billing_factor, 2)
    total_long_distance = round(
        avg_long_distance * tenure * clamp(rng.gauss(0.99, 0.04), 0.85, 1.10), 2
    )

    extra_data = 0
    if internet and not unlimited and avg_gb > 30:
        extra_data = 10 * min(15, max(0, int((avg_gb - 25) / 5 + rng.random() * 2)))

    refunds = round(rng.uniform(1, 50), 2) if rng.random() < 0.075 else 0.0
    total_revenue = round(max(0, total_charges + total_long_distance + extra_data - refunds), 2)

    lp = 0.0
    lp += {"Month-to-Month": 1.05, "One Year": 0.25, "Two Year": -0.35}[contract]
    lp += -0.020 * (tenure - 32)
    lp += -0.48 * (satisfaction - 3)
    lp += 0.28 * ((charge - 65) / 30)
    lp += 0.32 * int(internet_type == "Fiber Optic")
    lp += 0.28 * int(internet and not premium_support)
    lp += 0.20 * int(internet and not online_security)
    lp += 0.18 * int(payment == "Mailed Check")
    lp += -0.07 * min(referrals, 5) - 0.08 * min(dependents, 3)
    lp += 0.10 * int(age >= 65)
    lp += 0.45 * int(contract == "Month-to-Month" and tenure < 6)
    lp += 0.22 * int(internet_type == "Fiber Optic" and charge > 95)
    lp += 0.28 * int(satisfaction <= 2 and internet and not premium_support)
    lp += 0.010 * (month_index - 5.5)
    lp += rng.gauss(0, 0.35)

    row = {
        "Customer ID": f"SYN-{customer_number:06d}",
        "Snapshot Date": snapshot.isoformat(),
        "Age": age,
        "Gender": gender,
        "Married": married,
        "Number of Dependents": dependents,
        "Number of Referrals": referrals,
        "Tenure in Months": tenure,
        "Contract": contract,
        "Payment Method": payment,
        "Paperless Billing": paperless,
        "Phone Service": phone,
        "Multiple Lines": multiple,
        "Internet Service": internet,
        "Internet Type": internet_type,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection Plan": device_protection,
        "Premium Tech Support": premium_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Streaming Music": streaming_music,
        "Unlimited Data": unlimited,
        "Avg Monthly GB Download": avg_gb,
        "Avg Monthly Long Distance Charges": avg_long_distance,
        "Offer": offer,
        "Preperiod Satisfaction Score": satisfaction,
        "Monthly Charge": charge,
        "Total Charges": total_charges,
        "Total Extra Data Charges": extra_data,
        "Total Long Distance Charges": total_long_distance,
        "Total Refunds": refunds,
        "Total Revenue": total_revenue,
    }
    return row, lp


def realized_intercept(lps, uniforms, target_rate):
    target_count = round(len(lps) * target_rate)
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        count = sum(sigmoid(mid + lp) >= u for lp, u in zip(lps, uniforms, strict=True))
        if count < target_count:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic TELCO churn splits."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synthetic",
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    feature_rng = random.Random(SEED)
    outcome_rng = random.Random(SEED + 17)
    customer_number = 1

    for split, n_rows in SIZES.items():
        rows, lps, uniforms = [], [], []
        dates = split_dates(split)

        for i in range(n_rows):
            row, lp = generate_predictors(feature_rng, dates[i % len(dates)], customer_number)
            customer_number += 1
            rows.append(row)
            lps.append(lp)
            uniforms.append(outcome_rng.random())

        intercept = realized_intercept(lps, uniforms, TARGET_RATES[split])
        for row, lp, u in zip(rows, lps, uniforms, strict=True):
            row["Churn Next 30D"] = int(sigmoid(intercept + lp) >= u)

        with (out / f"telco_churn_synthetic_{split}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
