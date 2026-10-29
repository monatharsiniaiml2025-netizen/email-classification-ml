import re
import pandas as pd

def clean_text(text):
    """Lowercase text and remove URLs, email addresses, and non-alphanumeric punctuation."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def prepare_dataset(path):
    df = pd.read_csv(path)
    required = {"subject", "body", "category"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset must contain these columns: {', '.join(sorted(required))}")
    df = df[["subject", "body", "category"]].copy()
    for col in ["subject", "body", "category"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["category"] = df["category"].replace("", pd.NA)
    df = df.dropna(subset=["category"])
    df = df[(df["subject"] + df["body"]).str.strip() != ""]
    df = df.drop_duplicates(subset=["subject", "body", "category"]).reset_index(drop=True)
    df["text"] = (df["subject"] + " " + df["body"]).map(clean_text)
    return df
