# Pune Transit Complaint Intelligence Website

A runnable Streamlit prototype based on the uploaded NLP case study report.

## Features
- Complaint text submission and validation
- Service category, sentiment, and priority predictions
- TF-IDF features with Multinomial Naive Bayes, Logistic Regression, and Linear SVM
- Pune-area location matching (dictionary-based starter implementation)
- Session complaint history, filters, status updates, and CSV export
- Held-out test-set metrics and confusion matrices
- Safety-keyword priority guardrail and human-review reminders

## Run locally (Windows)
1. Install Python 3.10–3.12 (recommended).
2. Extract this ZIP to a folder.
3. Open PowerShell in that folder.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
5. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Start the website:
   ```powershell
   streamlit run app.py
   ```
7. Streamlit will show a local URL, usually http://localhost:8501.

## Important data note
`complaints_sample.csv` contains synthetic demonstration data. The model metrics are for demonstration only and must not be reported as real-world Pune transport performance. For a genuine project evaluation, replace it with an authorized, anonymized, manually labelled dataset containing these columns:
- `complaint`
- `category`
- `sentiment`
- `priority`

Allowed category examples in this prototype:
`Bus Delay`, `Overcrowding`, `Staff Behaviour`, `Ticketing Issue`, `Cleanliness`, `Safety Issue`, `Service Disruption`, `Other`

Sentiment labels: `Positive`, `Negative`, `Neutral`
Priority labels: `High`, `Medium`, `Low`

Keep a held-out test set, avoid data leakage, and report per-class precision/recall/F1 as well as macro and weighted averages. Do not use model output as a substitute for emergency response or authorized human judgment.

## Prototype limitations
- History exists only in the current Streamlit session.
- Location extraction uses a small built-in Pune place-name list.
- No login, permanent database, live PMPML API, or actual complaint submission is configured.
- English text only in this starter implementation.
