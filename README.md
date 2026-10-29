# AI-Based Email Classification System

A beginner-friendly end-to-end machine learning project that predicts an email department/category from its subject and body.

## Features
- Dataset cleaning: missing-value handling, duplicate removal, lowercase conversion, and special-character cleanup
- Text vectorization with TF-IDF and English stop-word removal
- Compares Logistic Regression and Multinomial Naive Bayes
- Reports accuracy, weighted precision, recall, F1-score, and confusion matrix
- Streamlit dashboard for predictions, confidence scores, prediction history, dataset overview, and evaluation
- Cleaned CSV, category chart, comparison table, classification report, and confusion matrix outputs

## Important dataset note
`data/emails.csv` contains a small **illustrative starter dataset created for demonstrating the pipeline**. It is not a Kaggle dataset and its metrics are not a real-world benchmark. For the final submission, download a permitted labeled dataset (for example, one of the datasets suggested in the task brief) and map its columns to:
- `subject`
- `body`
- `category`

Keep one email per row. Each category should have enough examples for a train/test split. Do not report starter-data scores as production performance.

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

If PowerShell blocks activation, run this only for the current terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Then open the local URL shown by Streamlit in your browser.

## Dataset format
Example CSV:
```csv
subject,body,category
Password reset request,I cannot access my account,IT Support
Invoice payment pending,The supplier invoice is overdue,Finance
Leave request,I need two days off,HR
```

If your downloaded dataset uses different column names, rename those columns in `data/emails.csv` before training. The current loader expects exactly `subject`, `body`, and `category`.

## Project structure
```text
email_classification_ml/
├── app.py
├── data/
│   └── emails.csv
├── models/                 # trained model and metadata created by training
├── reports/                # generated reports and charts
├── src/
│   ├── __init__.py
│   ├── train_model.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## How it works
1. Load and clean the labeled email dataset.
2. Combine subject and body into one text field.
3. Split the data into training and test sets with stratification.
4. Convert text into TF-IDF features.
5. Train and compare two classification algorithms.
6. Save the model with the best weighted F1 score.
7. Use the saved pipeline to classify new emails in the dashboard.

## Evaluation
The training script writes:
- `reports/category_distribution.csv`
- `reports/category_distribution.png`
- `reports/cleaned_emails.csv`
- `reports/model_comparison.csv`
- `reports/classification_report.txt`
- `reports/confusion_matrix.png`

## Limitations and future improvements
- Replace the illustrative dataset with a larger, representative, licensed dataset.
- Check for class imbalance and collect more examples for underrepresented classes.
- Tune hyperparameters and use cross-validation on the real dataset.
- Add priority prediction and department routing.
- Compare with a small language model if compute and dataset access permit.
- Avoid sending confidential email content to external services without authorization.
