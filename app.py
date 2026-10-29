
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

from src.utils import clean_text, prepare_dataset

# -------------------- PATHS --------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "email_classifier.joblib"
META_PATH = ROOT / "models" / "metadata.json"
DATA_PATH = ROOT / "data" / "emails.csv"
REPORT_PATH = ROOT / "reports"

# -------------------- PAGE CONFIGURATION --------------------
st.set_page_config(
    page_title="AI Email Classifier",
    page_icon="📧",
    layout="wide"
)

st.title("📧 AI-Based Email Classification System")
st.caption(
    "Classify emails into categories using Natural Language Processing "
    "(NLP) and Machine Learning."
)

# -------------------- CHECK MODEL --------------------
if not MODEL_PATH.exists():
    st.warning(
        "The model has not been trained yet. "
        "Run `python src/train_model.py` in the project folder."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()

metadata = {}
if META_PATH.exists():
    try:
        metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        metadata = {}

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("About this project")

    st.write(
        "TF-IDF text features + supervised Machine Learning "
        "for email classification."
    )

    st.write(
        f"Selected model: **{metadata.get('best_model', 'Logistic Regression')}**"
    )

    categories = metadata.get("categories", [])
    if categories:
        st.write("**Email categories:**")
        st.write(", ".join(categories))

    st.info(
        "The current dataset contains 2,104 email records "
        "across 10 categories. Predictions depend on email content "
        "and the quality of the training data."
    )

# -------------------- SESSION STATE --------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs(
    ["📩 Predict Email", "📊 Dataset Overview", "📈 Model Evaluation"]
)

# =====================================================
# TAB 1: EMAIL PREDICTION
# =====================================================
with tab1:
    st.subheader("Classify an Email")

    with st.form("email_form"):
        subject = st.text_input(
            "Email Subject",
            placeholder="Example: Interview invitation"
        )

        body = st.text_area(
            "Email Content",
            height=170,
            placeholder="Enter or paste the email content here..."
        )

        submitted = st.form_submit_button(
            "Classify Email",
            type="primary",
            width="stretch"
        )

    if submitted:
        if not subject.strip() and not body.strip():
            st.error("Please enter an email subject or email content.")
        else:
            text = clean_text(subject + " " + body)

            if not text:
                st.error("The email does not contain usable text.")
            else:
                try:
                    prediction = model.predict([text])[0]

                    confidence = None
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba([text])[0]
                        confidence = float(max(probabilities))

                    result = {
                        "subject": subject.strip() or "(No subject)",
                        "category": str(prediction),
                        "confidence": confidence
                    }

                    st.session_state.history.insert(0, result)

                    st.success(
                        f"Predicted category: {prediction}"
                    )

                    if confidence is not None:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric(
                                "Model Confidence",
                                f"{confidence:.1%}"
                            )

                        with col2:
                            st.metric(
                                "Predicted Category",
                                str(prediction)
                            )

                        st.caption(
                            "Confidence is the model's estimated probability, "
                            "not a guarantee that the prediction is correct."
                        )

                        if confidence < 0.40:
                            st.warning(
                                "The model has relatively low confidence "
                                "in this prediction. Please verify the category."
                            )

                except Exception as error:
                    st.error(f"Prediction failed: {error}")

    st.divider()
    st.subheader("Prediction History")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)

        history_df["confidence"] = history_df["confidence"].map(
            lambda value: f"{value:.1%}"
            if pd.notna(value)
            else "Not available"
        )

        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet. Classify an email to see its history.")

# =====================================================
# TAB 2: DATASET OVERVIEW
# =====================================================
with tab2:
    st.subheader("Dataset Analysis")

    if not DATA_PATH.exists():
        st.error(f"Dataset file not found: {DATA_PATH}")
    else:
        try:
            df = prepare_dataset(DATA_PATH)

            total_emails = len(df)
            total_categories = df["category"].nunique()

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Emails", f"{total_emails:,}")
            col2.metric("Total Categories", total_categories)
            col3.metric(
                "Missing Values",
                int(df[["subject", "body", "category"]].isna().sum().sum())
            )

            st.divider()
            st.subheader("Email Distribution by Category")

            category_counts = (
                df["category"]
                .value_counts()
                .rename_axis("Category")
                .reset_index(name="Email Count")
            )

            st.bar_chart(
                category_counts.set_index("Category")["Email Count"],
                width="stretch"
            )

            st.subheader("Category Counts")

            st.dataframe(
                category_counts,
                width="stretch",
                hide_index=True
            )

            st.subheader("Sample Email Records")

            st.dataframe(
                df[["subject", "body", "category"]].head(10),
                width="stretch",
                hide_index=True
            )

            st.caption(
                "This analysis is generated from the current data/emails.csv "
                "file after dataset cleaning."
            )

        except Exception as error:
            st.error(f"Unable to analyze the dataset: {error}")

# =====================================================
# TAB 3: MODEL EVALUATION
# =====================================================
with tab3:
    st.subheader("Model Evaluation Results")

    comparison_path = REPORT_PATH / "model_comparison.csv"
    confusion_matrix_path = REPORT_PATH / "confusion_matrix.png"
    classification_report_path = REPORT_PATH / "classification_report.txt"

    if comparison_path.exists():
        try:
            comparison_df = pd.read_csv(comparison_path)

            st.write("**Model Performance Comparison**")

            st.dataframe(
                comparison_df,
                width="stretch",
                hide_index=True
            )

            if "model" in comparison_df.columns:
                metric_columns = [
                    column
                    for column in [
                        "accuracy",
                        "weighted_f1",
                        "macro_f1"
                    ]
                    if column in comparison_df.columns
                ]

                if metric_columns:
                    st.subheader("Performance Metrics")

                    chart_df = comparison_df.set_index("model")[metric_columns]
                    st.bar_chart(chart_df, width="stretch")

        except Exception as error:
            st.error(f"Unable to load model comparison: {error}")
    else:
        st.info(
            "Model comparison is not available. "
            "Run `python src/train_model.py` first."
        )

    if confusion_matrix_path.exists():
        st.subheader("Confusion Matrix")
        st.image(
            str(confusion_matrix_path),
            caption="Confusion matrix generated during model training",
            width="stretch"
        )

    if classification_report_path.exists():
        st.subheader("Classification Report")

        report_text = classification_report_path.read_text(
            encoding="utf-8"
        )

        st.code(report_text, language="text")

    st.caption(
        "Evaluation metrics are based on the test split used during training. "
        "Actual performance may vary on new emails."
    )

# -------------------- FOOTER --------------------
st.divider()
st.caption(
    "AI-Based Email Classification System | "
    "Built with Python, Streamlit, Pandas and Scikit-learn"
)
