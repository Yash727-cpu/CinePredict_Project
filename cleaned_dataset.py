
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

raw_path = DATASET_DIR / "tmdb_5000_movies.csv"
output_path = DATASET_DIR / "cleaned_movies.csv"

# Load real movie dataset
df = pd.read_csv(raw_path)

print("Original dataset shape:", df.shape)

# Select required columns
columns = [
    "budget",
    "revenue",
    "popularity",
    "runtime",
    "vote_average",
    "vote_count",
    "release_date"
]

df = df[columns].copy()

# Convert numeric columns
numeric_columns = [
    "budget", "revenue", "popularity",
    "runtime", "vote_average", "vote_count"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Extract release year
df["release_date"] = pd.to_datetime(
    df["release_date"], errors="coerce"
)
df["release_year"] = df["release_date"].dt.year

# Remove rows without valid budget/revenue
# Zero values often represent unknown financial data
df = df.dropna(
    subset=numeric_columns + ["release_year"]
)

df = df[
    (df["budget"] > 0) &
    (df["revenue"] > 0) &
    (df["runtime"] > 0) &
    (df["vote_count"] >= 0)
].copy()

# Define financial success:
# 1 = Revenue at least twice the production budget
# 0 = Otherwise
df["success"] = (
    df["revenue"] >= 2 * df["budget"]
).astype(int)

# Select final training columns
final_columns = [
    "budget",
    "popularity",
    "runtime",
    "vote_average",
    "vote_count",
    "release_year",
    "success"
]

df = df[final_columns].copy()
df["release_year"] = df["release_year"].astype(int)

# Verify both target classes exist
if df["success"].nunique() != 2:
    raise ValueError("Dataset must contain both success classes.")

# Save cleaned dataset
df.to_csv(output_path, index=False)

print("\nDataset cleaned successfully!")
print("Cleaned dataset shape:", df.shape)
print("\nSuccess distribution:")
print(df["success"].value_counts())
print("\nSaved to:", output_path)
