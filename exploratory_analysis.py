import streamlit as st
import pandas as pd
import plotly.express as px

def show_eda(df):
    st.markdown("### Descriptive statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    st.markdown("### Class distribution")
    if "SLA_Breached" in df.columns:
        labeled = df[df["SLA_Breached"].notna()].copy()
        if not labeled.empty:
            labeled["SLA_Breached_Label"] = labeled["SLA_Breached"].map(
                {0.0: "Not Breached", 1.0: "SLA Breached", 0: "Not Breached", 1: "SLA Breached"}
            )
            counts = labeled["SLA_Breached_Label"].value_counts().rename_axis("Outcome").reset_index(name="Count")
            st.plotly_chart(px.pie(counts, names="Outcome", values="Count", title="SLA Outcome Distribution"), use_container_width=True)
        else:
            st.info("No completed records with known SLA outcomes are available.")
    else:
        st.info("SLA_Breached is created from Processing_Days > SLA_Days for completed records.")

    st.markdown("### Interactive charts")
    service_counts = df["Service_Type"].value_counts().rename_axis("Service Type").reset_index(name="Requests")
    st.plotly_chart(px.bar(service_counts, x="Service Type", y="Requests", title="Request Volume by Service Type"), use_container_width=True)

    if "SLA_Breached" in df.columns:
        done = df[df["SLA_Breached"].notna()].copy()
        if not done.empty:
            by_office = done.groupby("Assigned_Office", dropna=False).agg(
                Requests=("SLA_Breached", "size"),
                Breach_Rate=("SLA_Breached", "mean")
            ).reset_index()
            by_office["Breach Rate (%)"] = by_office["Breach_Rate"] * 100
            st.plotly_chart(px.bar(by_office, x="Assigned_Office", y="Breach Rate (%)", title="SLA Breach Rate by Assigned Office"), use_container_width=True)

            if "Queue_Category" in done.columns:
                by_queue = done.groupby("Queue_Category", observed=False).agg(
                    Requests=("SLA_Breached", "size"),
                    Breach_Rate=("SLA_Breached", "mean")
                ).reset_index()
                by_queue["Breach Rate (%)"] = by_queue["Breach_Rate"] * 100
                st.plotly_chart(px.bar(by_queue, x="Queue_Category", y="Breach Rate (%)", title="SLA Breach Rate by Queue Category"), use_container_width=True)
