# Student Service Request Monitoring System

Streamlit deployment project for the Section B laboratory exercise. It reuses the Decision Tree model and feature definitions from the uploaded midterm notebook.

## Files
- `app.py` — Streamlit application and navigation
- `modules/` — data validation, EDA, prediction, dashboard, and SAD pages
- `models/trained_model.joblib` — saved midterm model bundle
- `assets/use_case_diagram.png` — system use case diagram
- `student_service_requests_synthetic.csv` — supplied synthetic dataset

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push the complete folder contents to a GitHub repository. In Streamlit Community Cloud, create an app using the repository, branch `main`, and main file path `app.py`.

## Model compatibility
The supplied model was saved with scikit-learn 1.6.1. The requirements file pins `scikit-learn==1.6.1` to match the model's saved estimator version. Do not change the scikit-learn version unless the model is re-exported from the training notebook in that version.

## Important limitations
- The target `SLA_Breached` is derived for completed requests from `Processing_Days > SLA_Days`.
- The original midterm pipeline uses the features stored in the saved model bundle.
- Model predictions are decision-support outputs only. Authorized staff must review requests and make final decisions.
- Use synthetic or properly authorized data for public deployment. Do not upload sensitive student records.
