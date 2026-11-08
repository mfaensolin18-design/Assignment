import pandas as pd
import numpy as np

EXPECTED_COLUMNS = [
    "Request_ID", "Service_Type", "Requester_Type", "Submission_Date",
    "Requested_Channel", "Priority", "Assigned_Office", "Queue_Length",
    "Required_Documents_Complete", "Processing_Days", "SLA_Days",
    "Status", "Reopened", "Satisfaction_Score"
]

def validate_and_clean(df):
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if df.empty:
        raise ValueError("The uploaded CSV has no rows.")

    data = df.copy()
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].str.strip()

    capitalization_map = {
        "Priority": {"low": "Low", "normal": "Normal", "high": "High"},
        "Required_Documents_Complete": {"yes": "Yes", "no": "No"},
        "Reopened": {"yes": "Yes", "no": "No"},
        "Status": {
            "completed": "Completed", "pending": "Pending",
            "open": "Open", "cancelled": "Cancelled", "canceled": "Cancelled"
        }
    }
    for col, mapping in capitalization_map.items():
        data[col] = data[col].apply(
            lambda x: mapping.get(x.lower(), x) if isinstance(x, str) else x
        )

    data["Submission_Date"] = pd.to_datetime(data["Submission_Date"], errors="coerce")
    numeric_cols = ["Queue_Length", "Processing_Days", "SLA_Days", "Satisfaction_Score"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    duplicate_ids = int(data["Request_ID"].duplicated().sum())
    data = data.drop_duplicates(subset=["Request_ID"], keep="first").copy()

    negative_values = 0
    for col in ["Queue_Length", "Processing_Days", "SLA_Days"]:
        mask = data[col] < 0
        negative_values += int(mask.sum())
        data.loc[mask, col] = np.nan

    invalid_satisfaction = int(
        ((data["Satisfaction_Score"] < 1) | (data["Satisfaction_Score"] > 5)).sum()
    )
    data.loc[
        (data["Satisfaction_Score"] < 1) | (data["Satisfaction_Score"] > 5),
        "Satisfaction_Score"
    ] = np.nan

    invalid_dates = int(data["Submission_Date"].isna().sum())
    data["Submission_Weekday"] = data["Submission_Date"].dt.day_name()
    data["Queue_Category"] = pd.cut(
        data["Queue_Length"],
        bins=[-np.inf, 5, 15, np.inf],
        labels=["Low", "Medium", "High"]
    )

    if {"Processing_Days", "SLA_Days"}.issubset(data.columns):
        eligible = data["Processing_Days"].notna() & data["SLA_Days"].notna()
        data["SLA_Breached"] = np.nan
        data.loc[eligible, "SLA_Breached"] = (
            data.loc[eligible, "Processing_Days"] > data.loc[eligible, "SLA_Days"]
        ).astype(int)

    report = {
        "duplicate_ids": duplicate_ids,
        "invalid_dates": invalid_dates,
        "negative_values": negative_values,
        "invalid_satisfaction": invalid_satisfaction
    }
    return data, report
