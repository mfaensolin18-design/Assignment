import streamlit as st
import pandas as pd
from modules.data_processing import validate_and_clean
from modules.exploratory_analysis import show_eda
from modules.prediction import load_model_bundle, make_prediction_form
from modules.dashboard import show_dashboard
from modules.system_design import show_system_design

st.set_page_config(
    page_title="Student Service Request Monitoring",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def get_model_bundle():
    return load_model_bundle("models/trained_model.joblib")

st.sidebar.title("🎓 Student Service Monitoring")
page = st.sidebar.selectbox(
    "Navigate",
    ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"]
)

st.sidebar.caption("Section B | SLA Breach Monitoring")
st.sidebar.info("Predictions support staff review. They do not automatically approve or deny student services.")

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None

if page == "Home":
    st.title("Student Service Request Monitoring System")
    st.subheader("SLA Breach Prediction and Workload Dashboard")
    st.write(
        "This application uses the Decision Tree model from the midterm project "
        "to help authorized student-services staff identify requests that may need "
        "attention before or during service processing."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Project Dataset", "400 records")
    c2.metric("Prediction Target", "SLA_Breached")
    c3.metric("Model", "Decision Tree")
    st.markdown("### Available modules")
    st.markdown(
        "- **Dataset:** upload and validate the student service request CSV.\n"
        "- **Data Analysis:** descriptive statistics and interactive charts.\n"
        "- **Prediction:** enter request details and estimate SLA-breach risk.\n"
        "- **Dashboard:** review workload and requests flagged for human follow-up.\n"
        "- **System Design:** functional requirements, actors, and workflow."
    )
    st.info("Start by opening the Dataset page and uploading student_service_requests_synthetic.csv.")

elif page == "Dataset":
    st.title("Dataset Upload and Validation")
    uploaded = st.file_uploader("Upload student service request CSV", type=["csv"])
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            cleaned, report = validate_and_clean(raw)
            st.session_state.raw_df = raw
            st.session_state.clean_df = cleaned
            st.success("CSV uploaded and checked.")
            a, b, c = st.columns(3)
            a.metric("Rows", f"{len(raw):,}")
            b.metric("Columns", len(raw.columns))
            c.metric("Duplicate Request IDs", report["duplicate_ids"])
            st.markdown("### Data preview")
            st.dataframe(raw.head(20), use_container_width=True)
            st.markdown("### Data quality summary")
            q1, q2 = st.columns(2)
            with q1:
                st.write("Missing values (original upload)")
                st.dataframe(raw.isna().sum().rename("Missing").to_frame(), use_container_width=True)
            with q2:
                st.write("Cleaning actions")
                st.write(f"- Duplicate IDs removed: {report['duplicate_ids']}")
                st.write(f"- Invalid dates converted to missing: {report['invalid_dates']}")
                st.write(f"- Negative numeric values converted to missing: {report['negative_values']}")
                st.write(f"- Invalid satisfaction scores converted to missing: {report['invalid_satisfaction']}")
            st.markdown("### Cleaned data preview")
            st.dataframe(cleaned.head(20), use_container_width=True)
            st.download_button(
                "Download cleaned CSV",
                cleaned.to_csv(index=False).encode("utf-8"),
                file_name="student_service_requests_cleaned.csv",
                mime="text/csv"
            )
        except Exception as exc:
            st.error(f"Could not read or validate this CSV: {exc}")
    elif st.session_state.clean_df is not None:
        st.info("Showing the dataset uploaded earlier in this session.")
        st.dataframe(st.session_state.clean_df.head(20), use_container_width=True)
    else:
        st.warning("Please upload the CSV file to begin.")

elif page == "Data Analysis":
    st.title("Exploratory Data Analysis")
    if st.session_state.clean_df is None:
        st.warning("Upload the CSV first on the Dataset page.")
    else:
        show_eda(st.session_state.clean_df)

elif page == "Prediction":
    st.title("SLA Breach Prediction")
    try:
        bundle = get_model_bundle()
        if st.session_state.clean_df is not None:
            choices_df = st.session_state.clean_df
        else:
            choices_df = None
        make_prediction_form(bundle, choices_df)
    except Exception as exc:
        st.error("The saved model could not be loaded.")
        st.exception(exc)
        st.info(
            "This model was saved with scikit-learn 1.6.1. "
            "Use the supplied requirements.txt and redeploy/reinstall dependencies "
            "if a version compatibility error appears."
        )

elif page == "Dashboard":
    st.title("Decision-Support Dashboard")
    try:
        bundle = get_model_bundle()
        if st.session_state.clean_df is None:
            st.warning("Upload the CSV first on the Dataset page.")
        else:
            show_dashboard(st.session_state.clean_df, bundle)
    except Exception as exc:
        st.error("The saved model could not be loaded.")
        st.exception(exc)
        st.info("Check that models/trained_model.joblib exists and scikit-learn 1.6.1 is installed.")

elif page == "System Design":
    st.title("System Design and Analysis")
    show_system_design()
