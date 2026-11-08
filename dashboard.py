import streamlit as st
import pandas as pd

def show_dashboard(df, bundle):
    model = bundle["model"]
    features = bundle["features"]
    eligible = df.copy()
    missing = [c for c in features if c not in eligible.columns]
    if missing:
        st.error("Dataset is missing model features: " + ", ".join(missing))
        return

    # Only use predictor columns from the original midterm pipeline.
    X = eligible[features].copy()
    predictions = model.predict(X)
    eligible["Predicted_SLA_Breached"] = predictions.astype(int)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = list(model.classes_)
        if 1 in classes:
            eligible["Breach_Probability"] = probabilities[:, classes.index(1)]
        else:
            eligible["Breach_Probability"] = 0.0
    else:
        eligible["Breach_Probability"] = pd.NA

    total = len(eligible)
    flagged = int((eligible["Predicted_SLA_Breached"] == 1).sum())
    pending = int(eligible["Status"].astype(str).str.lower().eq("pending").sum()) if "Status" in eligible else 0
    avg_queue = pd.to_numeric(eligible["Queue_Length"], errors="coerce").mean()

    a, b, c, d = st.columns(4)
    a.metric("Total Requests", f"{total:,}")
    b.metric("Flagged for Review", f"{flagged:,}")
    c.metric("Pending Requests", f"{pending:,}")
    d.metric("Average Queue Length", "N/A" if pd.isna(avg_queue) else f"{avg_queue:.1f}")

    st.markdown("### Staff review queue")
    flagged_df = eligible[eligible["Predicted_SLA_Breached"] == 1].copy()
    sort_cols = ["Breach_Probability"] if flagged_df["Breach_Probability"].notna().any() else []
    if sort_cols:
        flagged_df = flagged_df.sort_values(sort_cols, ascending=False)
    display_cols = [c for c in [
        "Request_ID", "Service_Type", "Assigned_Office", "Priority",
        "Queue_Length", "SLA_Days", "Status", "Predicted_SLA_Breached",
        "Breach_Probability"
    ] if c in flagged_df.columns]
    st.dataframe(flagged_df[display_cols], use_container_width=True)
    if not flagged_df.empty:
        st.download_button(
            "Download staff review queue",
            flagged_df.to_csv(index=False).encode("utf-8"),
            file_name="sla_review_queue.csv",
            mime="text/csv"
        )
    st.info("Recommended human follow-up: verify request details, check missing documents, contact the assigned office, and record the staff member's final action. Do not automatically reject or deny student services.")
    st.caption("Predictions are based on the saved midterm Decision Tree pipeline and the selected input features.")
