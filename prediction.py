import joblib
import pandas as pd
import streamlit as st

@st.cache_resource
def load_model_bundle(model_path):
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError("The saved joblib file must contain a dictionary with 'model' and 'features'.")
    return bundle

def _options(df, col, defaults):
    if df is not None and col in df.columns:
        values = df[col].dropna().astype(str).unique().tolist()
        if values:
            return sorted(values)
    return defaults

def make_prediction_form(bundle, df=None):
    model = bundle["model"]
    features = bundle["features"]
    st.write("Enter the request details. The input fields match the features used in the midterm model.")
    defaults = {
        "Service_Type": ["Academic Records", "Certification", "IT Support", "Scholarship Inquiry", "Enrollment", "Financial Assistance"],
        "Requester_Type": ["Student", "Faculty", "Staff"],
        "Submission_Weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "Requested_Channel": ["Portal", "Email", "Walk-in"],
        "Priority": ["Low", "Normal", "High"],
        "Assigned_Office": ["Registrar", "IT Services", "Scholarship Office", "Student Affairs", "Finance Office"],
        "Queue_Category": ["Low", "Medium", "High"],
        "Required_Documents_Complete": ["Yes", "No"]
    }
    numeric = {"Queue_Length": (0, 0, 100), "SLA_Days": (3, 1, 30)}
    with st.form("prediction_form"):
        cols = st.columns(2)
        values = {}
        for idx, feature in enumerate(features):
            with cols[idx % 2]:
                if feature in numeric:
                    default, minimum, maximum = numeric[feature]
                    values[feature] = st.number_input(
                        feature.replace("_", " "), min_value=minimum,
                        max_value=maximum, value=default, step=1
                    )
                else:
                    opts = _options(df, feature, defaults.get(feature, ["Unknown"]))
                    values[feature] = st.selectbox(feature.replace("_", " "), opts)
        submitted = st.form_submit_button("Predict SLA Risk", type="primary")

    if submitted:
        input_df = pd.DataFrame([{f: values[f] for f in features}], columns=features)
        prediction = int(model.predict(input_df)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            classes = list(model.classes_)
            if 1 in classes:
                probability = float(probs[classes.index(1)])
        if prediction == 1:
            st.error("Prediction: SLA Breach Risk")
        else:
            st.success("Prediction: Not flagged as an SLA breach")
        if probability is not None:
            st.metric("Estimated SLA breach probability", f"{probability:.1%}")
        st.warning("This is a model estimate, not a final decision. Authorized service staff must review the request and its context.")
        st.caption("Model target: SLA_Breached | Class 0 = Not Breached | Class 1 = SLA Breached")
