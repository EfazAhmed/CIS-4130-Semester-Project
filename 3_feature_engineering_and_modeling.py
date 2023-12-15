# ================================================================================================== #
# Import the necessary libraries for this milestone
# ================================================================================================== #
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import sklearn
import numpy
import scipy
import plotly
import bs4 as bs
import urllib.request
import boto3
import os
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql.types import StringType
from pyspark.sql.functions import *
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    HashingTF,
    IDF,
    Tokenizer,
    RegexTokenizer,
)
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import *
from pyspark.ml.tuning import *
import numpy as np
from textblob import TextBlob


# ================================================================================================== #
# Prepared environment variables and variables save paths and filenames
# ================================================================================================== #

access_key = "---INSERT ACCESS KEY---"
secret_key = "---INSERT SECRET KEY---"
os.environ["AWS_ACCESS_KEY_ID"] = access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
aws_region = "us-east-2"

sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", access_key)
sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", secret_key)
sc._jsc.hadoopConfiguration().set(
    "fs.s3a.endpoint", "s3." + aws_region + ".amazonaws.com"
)

bucket_path = "s3://amazon-reviews-ea/"
bucket_name = "amazon-reviews-ea"

s3_client = boto3.client("s3")
response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="landing/")
objects = response.get("Contents")

filenames = [obj["Key"] for obj in objects][1:]


# ================================================================================================== #
# Cleaned the data in the landing directory from the Amazon S3 bucket using PySpark and DataBricks
# and saved the clean data in a parquet file in the raw directory
# ================================================================================================== #


@udf
def ascii_only(text):
    return text.encode("ascii", "ignore").decode("ascii") if text else None


for i, filename in enumerate(filenames):
    input_file_path = bucket_path + filename
    output_file_path = (
        f"s3://amazon-reviews-ea/raw/cleaned_{filename[8:]}"[:-3] + "parquet"
    )
    print(f"{i+1}/{len(filenames)} {input_file_path} ---> {output_file_path}")

    sdf = spark.read.csv(input_file_path, sep="\t", header=True, inferSchema=True)

    # Select columns
    cols = [
        "marketplace",
        "product_title",
        "product_category",
        "star_rating",
        "helpful_votes",
        "total_votes",
        "verified_purchase",
        "review_headline",
        "review_body",
        "review_date",
    ]
    sdf = sdf.select(cols)

    # Clean text-based columns
    sdf = sdf.withColumn("review_headline", ascii_only("review_headline"))
    sdf = sdf.withColumn("review_body", ascii_only("review_body"))

    # Remove null values
    sdf = sdf.na.drop(subset=["star_rating", "review_body"])

    # Drop duplicates
    sdf = sdf.dropDuplicates()

    # Save the file in Amazon S3
    sdf.write.parquet(output_file_path)


# ================================================================================================== #
# Aggregated all of the parquet files into a singular PySpark DataFrame (main_sdf)
# ================================================================================================== #

response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="raw/")
objects = response.get("Contents")
filenames = list(set([obj["Key"][: obj["Key"].find(".") + 8] for obj in objects][1:]))

main_sdf = None

for i, filename in enumerate(filenames):
    input_file_path = bucket_path + filename
    print(f"{i+1}/{len(filenames)} {input_file_path}")

    sdf = spark.read.parquet(input_file_path)

    if not main_sdf:
        main_sdf = sdf
    else:
        main_sdf = reduce(DataFrame.unionAll, [main_sdf, sdf])

main_sdf = main_sdf.sample(False, 0.01)


# ================================================================================================== #
# Created and applied the model pipeline
# ================================================================================================== #

indexer_1 = StringIndexer(
    inputCol="product_category", outputCol="product_category_index"
)
regexTokenizer_1 = RegexTokenizer(
    inputCol="review_body", outputCol="review_body_tokens", pattern="\\w+", gaps=False
)
regexTokenizer_2 = RegexTokenizer(
    inputCol="review_headline",
    outputCol="review_headline_tokens",
    pattern="\\w+",
    gaps=False,
)
regexTokenizer_3 = RegexTokenizer(
    inputCol="product_title",
    outputCol="product_title_tokens",
    pattern="\\w+",
    gaps=False,
)
pipeline = Pipeline(
    stages=[indexer_1, regexTokenizer_1, regexTokenizer_2, regexTokenizer_3]
)
main_sdf = pipeline.fit(main_sdf).transform(main_sdf)


# ================================================================================================== #
# Conducted more feature engineering with additional encoding, sentiment analysis,
# and feature extraction from text-based features
# ================================================================================================== #


@udf
def sentiment_analysis(text):
    sentiment = TextBlob(text).sentiment.polarity
    return sentiment


main_sdf = main_sdf.withColumn(
    "verified_purchase", (col("verified_purchase") == "Y").cast("int")
)
main_sdf = main_sdf.withColumn(
    "sentiment_score", sentiment_analysis(col("review_body"))
)
main_sdf = main_sdf.withColumn(
    "review_headline_word_count", get_word_count(col("review_headline"))
)
main_sdf = main_sdf.withColumn(
    "review_body_word_count", get_word_count(col("review_body"))
)
main_sdf = main_sdf.drop(
    *["review_headline", "review_body", "product_title", "product_category"]
)


# ================================================================================================== #
# Casted variables to appropriate data types for the VectorAssembler
# ================================================================================================== #

main_sdf = main_sdf.withColumn(
    "review_headline_word_count", main_sdf.review_headline_word_count.cast("double")
)
main_sdf = main_sdf.withColumn(
    "review_body_word_count", main_sdf.review_body_word_count.cast("double")
)
main_sdf = main_sdf.withColumn(
    "sentiment_score", main_sdf.sentiment_score.cast("double")
)
main_sdf = main_sdf.withColumn("star_rating", main_sdf.star_rating.cast("double"))


# ================================================================================================== #
# Used hashingTF and IDF to encode text-based tokens
# ================================================================================================== #

hashingTF = HashingTF(
    numFeatures=4096, inputCol="review_body_tokens", outputCol="review_body_hashed"
)
main_sdf = hashingTF.transform(main_sdf)
idfModel = IDF(
    inputCol="review_body_hashed", outputCol="review_body_features", minDocFreq=1
).fit(main_sdf)
main_sdf = idfModel.transform(main_sdf)
hashingTF = HashingTF(
    numFeatures=4096,
    inputCol="review_headline_tokens",
    outputCol="review_headline_hashed",
)
main_sdf = hashingTF.transform(main_sdf)
idfModel = IDF(
    inputCol="review_headline_hashed",
    outputCol="review_headline_features",
    minDocFreq=1,
).fit(main_sdf)
main_sdf = idfModel.transform(main_sdf)
hashingTF = HashingTF(
    numFeatures=4096, inputCol="product_title_tokens", outputCol="product_title_hashed"
)
main_sdf = hashingTF.transform(main_sdf)
idfModel = IDF(
    inputCol="product_title_hashed", outputCol="product_title_features", minDocFreq=1
).fit(main_sdf)
main_sdf = idfModel.transform(main_sdf)
main_sdf = main_sdf.drop(
    *["review_body_tokens", "review_headline_tokens", "product_title_tokens"]
)


# ================================================================================================== #
# Created and evaluated the model, and uploaded the data and model to Amazon S3
# ================================================================================================== #

input_cols = [el for el in main_sdf.columns if el != "star_rating"]
assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
main_sdf = assembler.transform(main_sdf)
main_sdf = main_sdf.withColumn(
    "label", when(col("star_rating") > 3, 1.0).otherwise(0.0)
)
sdf = main_sdf.select(["features", "label"])
sdf.show()
# Upload data to "/trusted" directory in Amazon S3
main_sdf.write.parquet("s3://amazon-reviews-ea/trusted/amazon_reviews_data.parquet")

# Split the data into 70% training and 30% test sets
trainingData, testData = sdf.randomSplit([0.7, 0.3], seed=42)
# Create a LogisticRegression Estimator
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Fit the model to the training data
model = lr.fit(trainingData)

# Upload model to "/model" directory in Amazon S3
model.save("s3://amazon-reviews-ea/model")

# Show model coefficients and intercept
print("Coefficients: ", model.coefficients)
print("Intercept: ", model.intercept)
# Test the model on the testData
test_results = model.transform(testData)

# Show the test results
print(predictions.select("probability", "prediction").show())

# Save the confusion matrix
cm = test_results.groupby("label").pivot("prediction").count().fillna(0).collect()


def calculate_recall_precision(cm):
    tn = cm[0][1]  # True Negative
    fp = cm[0][2]  # False Positive
    fn = cm[1][1]  # False Negative
    tp = cm[1][2]  # True Positive
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    return accuracy, precision, recall, f1_score


print(calculate_recall_precision(cm))


# ================================================================================================== #
# Use the Cross-Validator to find the best model
# ================================================================================================== #

# Create a BinaryClassificationEvaluator to evaluate how well the model works
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

# Create the parameter grid (empty for now)
grid = ParamGridBuilder().build()

# Create the CrossValidator
cv = CrossValidator(
    estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, numFolds=3
)

# Use the CrossValidator to Fit the training data
cv = cv.fit(trainingData)

# Show the average performance over the three folds
cv.avgMetrics

# Evaluate the test data using the cross-validator model
# Reminder: We used Area Under the Curve
evaluator.evaluate(cv.transform(testData))

# Create a grid to hold hyperparameters
grid = ParamGridBuilder()
grid = grid.addGrid(lr.regParam, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
grid = grid.addGrid(lr.elasticNetParam, [0, 1])

# Build the grid
grid = grid.build()
print("Number of models to be tested: ", len(grid))

# Create the CrossValidator using the new hyperparameter grid
cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)

# Call cv.fit() to create models with all of the combinations of parameters in the grid
all_models = cv.fit(trainingData)
print("Average Metrics for Each model: ", all_models.avgMetrics)

model = all_models.bestModel
print("Area under ROC curve:", model.summary.areaUnderROC)  # ~0.87
