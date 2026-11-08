import streamlit as st

def show_system_design():
    st.markdown("### System purpose")
    st.write("The system helps student service staff monitor request workload and identify requests that may require SLA follow-up. Predictions are advisory and require authorized human review.")

    st.markdown("### Actors")
    st.markdown("- **Student / Requester:** submits a service request and views request status.\n- **Service Staff:** manages requests, monitors queues, reviews flagged requests, and follows up.\n- **Administrator:** manages accounts, monitors system use, and reviews reports.")

    st.markdown("### Five functional requirements")
    requirements = [
        "The system shall allow an authorized user to upload a student service request CSV file.",
        "The system shall validate required columns and summarize missing values, duplicate IDs, and invalid values.",
        "The system shall display descriptive statistics and at least three interactive data visualizations.",
        "The system shall load the saved Decision Tree pipeline and predict the SLA_Breached class for a new request using the original model features.",
        "The system shall display a review queue and allow authorized staff to download flagged requests for human follow-up."
    ]
    for i, req in enumerate(requirements, 1):
        st.write(f"{i}. {req}")

    st.markdown("### Three non-functional requirements")
    st.markdown("- **Security:** use synthetic or authorized data; do not publish passwords or sensitive student records.\n- **Usability:** provide clear navigation, labels, validation messages, and readable charts.\n- **Reliability:** handle invalid CSV files and model-loading errors with clear messages.")

    st.markdown("### Decision workflow")
    st.code(
        "Upload CSV → Validate schema and data → Prepare model features → "
        "Run saved Decision Tree → Display prediction and risk queue → "
        "Authorized staff reviews evidence → Staff records final action",
        language="text"
    )
    st.markdown("### Human decision and access")
    st.write("Students may submit requests and view their own request status. Service staff may view workload and model flags needed for their duties. Administrators manage access and system configuration. Only authorized staff make final service decisions.")
    st.warning("A model prediction must never be the sole reason to reject a reservation or deny student services.")
    st.markdown("### Updated use case diagram")
    st.image("assets/use_case_diagram.png", caption="Student Service Request Monitoring System — use case diagram", use_container_width=True)
