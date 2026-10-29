import pandas as pd

df = pd.read_csv("data/emails.csv")

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df[["subject", "body", "category"]].isnull().sum())

print("\nCategory counts:")
print(df["category"].value_counts())

print("\nRows where category differs from first all_labels value:")
if "all_labels" in df.columns:
    first_label = df["all_labels"].fillna("").str.split("|").str[0].str.strip()
    mismatch = df[first_label != df["category"].fillna("").str.strip()]
    print(mismatch[["subject", "category", "all_labels"]].head(20).to_string(index=False))