import pandas as pd


def validate_and_clean(df: pd.DataFrame):
    """
    Validate and clean a student service request dataset.

    Returns:
        cleaned_df: cleaned copy of the uploaded DataFrame
        report: dictionary of cleaning counts used by the Streamlit app

    Expected common columns (if present):
        Request_ID, Request_Date, Satisfaction_Score

    The function preserves other columns so it can work with the feature names
    used by your trained model.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The uploaded CSV has no records.")

    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    # Remove completely empty rows and columns.
    cleaned = cleaned.dropna(how="all").dropna(axis=1, how="all")

    # Normalize blank text cells to missing values.
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].replace(r"^\s*$", pd.NA, regex=True)

    # Find a request ID column, allowing common naming variations.
    id_candidates = [
        "Request_ID", "Request ID", "request_id",
        "RequestID", "Ticket_ID", "Ticket ID"
    ]
    id_column = next((name for name in id_candidates if name in cleaned.columns), None)

    duplicate_ids = 0
    if id_column is not None:
        valid_ids = cleaned[id_column].notna()
        duplicate_mask = valid_ids & cleaned[id_column].duplicated(keep="first")
        duplicate_ids = int(duplicate_mask.sum())
        cleaned = cleaned.loc[~duplicate_mask].copy()

    # Convert columns whose names indicate dates or timestamps.
    invalid_dates = 0
    date_keywords = ("date", "time", "timestamp")
    for column in cleaned.columns:
        if any(word in str(column).lower() for word in date_keywords):
            original = cleaned[column]
            converted = pd.to_datetime(original, errors="coerce")
            invalid_dates += int((original.notna() & converted.isna()).sum())
            cleaned[column] = converted

    # Convert numeric-looking columns, excluding identifiers and date fields.
    negative_values = 0
    id_keywords = ("id", "phone", "contact", "year")
    for column in cleaned.columns:
        if pd.api.types.is_datetime64_any_dtype(cleaned[column]):
            continue
        if any(word in str(column).lower() for word in id_keywords):
            continue

        name = str(column).lower()
        numeric_hint = any(
            word in name for word in (
                "day", "hour", "duration", "age", "score",
                "rating", "amount", "count", "number"
            )
        )

        if pd.api.types.is_numeric_dtype(cleaned[column]) or numeric_hint:
            converted = pd.to_numeric(cleaned[column], errors="coerce")
            negative_mask = converted < 0
            negative_values += int(negative_mask.sum())
            converted = converted.mask(negative_mask)
            cleaned[column] = converted

    # Satisfaction scores are expected to be on a 1-to-5 scale.
    invalid_satisfaction = 0
    satisfaction_column = next(
        (
            column for column in cleaned.columns
            if str(column).strip().lower().replace(" ", "_")
            in {"satisfaction_score", "satisfaction", "satisfaction_rating"}
        ),
        None
    )

    if satisfaction_column is not None:
        scores = pd.to_numeric(cleaned[satisfaction_column], errors="coerce")
        invalid_mask = scores.notna() & ~scores.between(1, 5)
        invalid_satisfaction = int(invalid_mask.sum())
        cleaned[satisfaction_column] = scores.mask(invalid_mask)

    # Reset row numbering after removals.
    cleaned = cleaned.reset_index(drop=True)

    report = {
        "duplicate_ids": duplicate_ids,
        "invalid_dates": invalid_dates,
        "negative_values": negative_values,
        "invalid_satisfaction": invalid_satisfaction,
    }

    return cleaned, report
