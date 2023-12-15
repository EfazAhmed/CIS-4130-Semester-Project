import json
import boto3
import pandas as pd
from IPython.display import HTML, display
from datetime import datetime

# Setting variables for the bucket path, filenames, and columns
bucket_path = "s3://amazon-reviews-ea/landing/"
file = open("kaggle_filenames.txt", "r")
filenames = json.load(file)

for filename in filenames:
    path = bucket_path + filename
    print(f"{'='*120}\nFile: {path}")
    df = pd.read_csv(
        path, usecols=cols, sep="\t", error_bad_lines=False, low_memory=False
    )

    # Encoded 'vine' and 'verified_purchase' columns for analysis
    df["vine"] = df["vine"].apply(lambda x: 1 if x == "Y" else 0 if x == "N" else None)
    df["verified_purchase"] = df["verified_purchase"].apply(
        lambda x: 1 if x == "Y" else 0
    )

    # Counting number of words in the 'review_headline' and 'review_body' columns
    df["review_headline_word_count"] = df["review_headline"].apply(
        lambda headline: len(str(headline).split())
    )
    df["review_body_word_count"] = df["review_body"].apply(
        lambda body: len(str(body).split())
    )

    # Selecting columns I am interested in analyzing
    num_cols = [
        "star_rating",
        "helpful_votes",
        "total_votes",
        "vine",
        "verified_purchase",
        "review_headline_word_count",
        "review_body_word_count",
    ]
    text_cols = ["review_headline", "review_body"]
    date_col = "review_date"

    # Changed 'review_date' type to datetime 'star_rating' type to be float64
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce")

    # Printing output for analysis
    print(f"List of Variables: {list(df.columns)}")
    print(f"Number of Observations: {df.shape[0]}")
    print(f"Number of Customers: {len(df['customer_id'].unique())}")
    print(f"Number of Products: {len(df['product_id'].unique())}")
    print(
        f"Number of Duplicates: {df.duplicated(subset='review_id', keep='first').sum()}"
    )
    print(f"Number of Missing/Null Values: {df.isna().sum().sum()}")
    print(f"Number of Missing/Null Values per Affected Column:")
    display(
        pd.DataFrame(
            df[df.columns[df.isnull().any()].to_list()].isna().sum(), columns=["count"]
        ).T
    )
    print(f"Min Review Date: {min(df[date_col])}")
    print(f"Max Review Date: {max(df[date_col])}")
    print(f"Min/Max/Avg/Stdev for Numerical Variables:")
    display(df[num_cols].agg(["min", "max", "mean", "std"]).round(decimals=3))
print("=" * 120)
