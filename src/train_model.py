
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils import prepare_dataset


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "emails.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Main training function
# --------------------------------------------------

def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # Load and clean dataset
    df = prepare_dataset(DATA_PATH)

    if df.empty:
        raise ValueError(
            "Dataset is empty after cleaning."
        )

    if df["category"].nunique() < 2:
        raise ValueError(
            "At least two categories are required."
        )

    counts = df["category"].value_counts()

    if counts.min() < 2:
        raise ValueError(
            "Each category needs at least two examples "
            "for a stratified train/test split."
        )

    # --------------------------------------------------
    # 3. Dataset analysis reports
    # --------------------------------------------------

    # Save cleaned dataset
    df.drop(columns=["text"]).to_csv(
        REPORT_DIR / "cleaned_emails.csv",
        index=False,
    )

    # Save category distribution CSV
    counts.rename_axis("category").reset_index(
        name="count"
    ).to_csv(
        REPORT_DIR / "category_distribution.csv",
        index=False,
    )

    # Save category distribution chart
    plt.figure(figsize=(10, 6))

    counts.sort_values().plot(
        kind="barh"
    )

    plt.title("Email Samples by Category")
    plt.xlabel("Number of Emails")
    plt.ylabel("Category")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "category_distribution.png",
        dpi=160,
    )

    plt.close()

    # Save updated dataset analysis text report
    analysis_lines = [
        "Dataset Analysis",
        "",
        f"Source: {DATA_PATH}",
        f"Total email rows after cleaning: {len(df)}",
        f"Number of categories: {df['category'].nunique()}",
        "",
        "Samples per category:",
    ]

    for category, count in counts.items():
        analysis_lines.append(
            f"- {category}: {count}"
        )

    missing_values = int(
        df[["subject", "body", "category"]]
        .isna()
        .sum()
        .sum()
    )

    duplicate_count = int(
        df.duplicated(
            subset=["subject", "body", "category"]
        ).sum()
    )

    analysis_lines.extend([
        "",
        f"Missing values after cleaning: {missing_values}",
        f"Duplicate rows after cleaning: {duplicate_count}",
        "",
        (
            "Note: Model performance depends on dataset quality, "
            "category distribution, and the test split. "
            "These results are not a real-world benchmark."
        ),
    ])

    (
        REPORT_DIR / "dataset_analysis.txt"
    ).write_text(
        "\n".join(analysis_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 4. Split dataset
    # --------------------------------------------------

    X = df["text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("Dataset loaded successfully")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Categories: {y.nunique()}")

    # --------------------------------------------------
    # 5. Define ML models
    # --------------------------------------------------

    candidates = {
        "Logistic Regression": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=12000,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]),

        "Multinomial Naive Bayes": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=12000,
                ),
            ),
            (
                "model",
                MultinomialNB(),
            ),
        ]),
    }

    # --------------------------------------------------
    # 6. Train and evaluate models
    # --------------------------------------------------

    results = []
    trained = {}

    for name, model in candidates.items():

        print(f"\nTraining: {name}")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        )

        macro_precision, macro_recall, macro_f1, _ = (
            precision_recall_fscore_support(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        results.append({
            "model": name,
            "accuracy": accuracy_score(
                y_test, predictions
            ),
            "weighted_precision": precision,
            "weighted_recall": recall,
            "weighted_f1": f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        })

        trained[name] = model

    # --------------------------------------------------
    # 7. Compare models
    # --------------------------------------------------

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        "weighted_f1",
        ascending=False,
    ).reset_index(drop=True)

    comparison.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\nModel Comparison:")
    print(comparison.to_string(index=False))

    # Select model with highest weighted F1-score
    best_name = comparison.iloc[0]["model"]
    best_model = trained[best_name]

    # Save selected model
    joblib.dump(
        best_model,
        MODEL_DIR / "email_classifier.joblib",
    )

    # --------------------------------------------------
    # 8. Classification report
    # --------------------------------------------------

    test_predictions = best_model.predict(X_test)

    report = classification_report(
        y_test,
        test_predictions,
        labels=sorted(y.unique()),
        zero_division=0,
    )

    report_text = (
        f"Best model: {best_name}\n"
        f"Test samples: {len(y_test)}\n"
        f"Training samples: {len(y_train)}\n\n"
        f"{report}\n\n"
        "IMPORTANT: These results depend on the supplied "
        "dataset and test split. They are not a "
        "real-world performance benchmark.\n"
    )

    (
        REPORT_DIR / "classification_report.txt"
    ).write_text(
        report_text,
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 9. Confusion matrix
    # --------------------------------------------------

    labels = sorted(y.unique())

    fig, ax = plt.subplots(figsize=(10, 8))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_predictions,
        labels=labels,
        xticks_rotation=45,
        cmap="Blues",
        ax=ax,
    )

    ax.set_title(
        f"Confusion Matrix - {best_name}"
    )

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR / "confusion_matrix.png",
        dpi=160,
    )

    plt.close(fig)

    # --------------------------------------------------
    # 10. Save model metadata
    # --------------------------------------------------

    metadata = {
        "best_model": best_name,
        "selection_metric": "weighted_f1",
        "categories": labels,
        "total_rows_after_cleaning": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "dataset_note": (
            "Results depend on the supplied dataset and "
            "test split. Not a real-world benchmark."
        ),
    }

    (
        MODEL_DIR / "metadata.json"
    ).write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 11. Completion message
    # --------------------------------------------------

    print("\nTraining completed successfully!")

    print(f"Best model: {best_name}")

    print(
        f"Saved model to: "
        f"{MODEL_DIR / 'email_classifier.joblib'}"
    )

    print(
        f"Reports saved to: {REPORT_DIR}"
    )


# --------------------------------------------------
# 12. Run program
# --------------------------------------------------

if __name__ == "__main__":
    main()
