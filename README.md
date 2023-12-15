![Amazon Logo](https://media.licdn.com/dms/image/C5612AQEHPbQPrqac6g/article-cover_image-shrink_423_752/0/1520222968224?e=1707955200&v=beta&t=Tnh6FQUxOrJ7EW2GMfWbV2jeVLgIy3Z_SZoQwJG1Adg)

**Course:** CIS 4130 (Big Data Technologies)
**Project Title:** Amazon US Customer Reviews
**Technologies**: Amazon EC2, Amazon S3, DataBricks, Python, PySpark, etc.

## Table of Contents
1. [Proposal](#Proposal)
2. [Data Aquisition](#Data-Aquisition)
3. [Exploratory Data Analysis (EDA)](#Exploratory-Data-Analysis)
4. [Feature Engineering and Modeling](#Feature-Engineering-and-Modeling)
5. [Data Visualizing](#Data-Visualizing)
6. [Summary and Conclusions](#Summary-and-Conclusions)

## Proposal
The dataset I intend to use is the Amazon US Customer Reviews Dataset. This dataset was found in Kaggle (22GB) and Hugging Face (32GB). The data source that I will be using will depend on which dataset holds more precedence in my analysis. For context, this dataset represents a sample of customer evaluations and opinions for different Amazon products (products listed on their website). There are over 130+ million customer reviews within the release of this dataset. This dataset was made to provide people of interest with a rich source of data for projects relating to natural language processing, machine learning, and many others. This dataset is divided into approximately 40 different product categories. Each category has features that are relevant to an Amazon product review. This includes the review body, the date of the review, and the star rating for the review. Each category also holds data on the product title as well as the product category. There are other attributes that exist for each category, but these are the main attributes that I will be focusing on in my analysis. I intend to conduct an analysis on the correlation of certain words to the star rating of the product. This will provide great insights on what words correlate with certain star ratings. I also intend to analyze trends between different product categories to see if any correlation exists between their sentiments.

**Objective:** The objective of this project was to create a machine learning pipeline that can predict if the star rating of an Amazon product is greater than 3 based on its various features using popular cloud technologies such as Amazon EC2, Amazon S3, and DataBricks.

- Kaggle: https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset 
- Hugging Face: https://huggingface.co/datasets/polinaeterna/amazon_us_reviews 

## Data Aquisition
1. Created an AWS EC2 instance and an AWS S3 Bucket
2. Connected into the AWS EC2 instance via EC2 Instance Connect
3. Created kaggle_filenames.json (nano kaggle_filenames.json) and inserted the following:

```
["amazon_reviews_multilingual_US_v1_00.tsv", "amazon_reviews_us_Apparel_v1_00.tsv", "amazon_reviews_us_Automotive_v1_00.tsv", "amazon_reviews_us_Baby_v1_00.tsv", "amazon_reviews_us_Beauty_v1_00.tsv", "amazon_reviews_us_Books_v1_02.tsv", "amazon_reviews_us_Camera_v1_00.tsv", "amazon_reviews_us_Digital_Ebook_Purchase_v1_01.tsv", "amazon_reviews_us_Digital_Music_Purchase_v1_00.tsv", "amazon_reviews_us_Digital_Software_v1_00.tsv", "amazon_reviews_us_Digital_Video_Download_v1_00.tsv", "amazon_reviews_us_Digital_Video_Games_v1_00.tsv", "amazon_reviews_us_Electronics_v1_00.tsv", "amazon_reviews_us_Furniture_v1_00.tsv", "amazon_reviews_us_Gift_Card_v1_00.tsv", "amazon_reviews_us_Grocery_v1_00.tsv", "amazon_reviews_us_Health_Personal_Care_v1_00.tsv", "amazon_reviews_us_Major_Appliances_v1_00.tsv", "amazon_reviews_us_Mobile_Apps_v1_00.tsv", "amazon_reviews_us_Mobile_Electronics_v1_00.tsv", "amazon_reviews_us_Musical_Instruments_v1_00.tsv", "amazon_reviews_us_Music_v1_00.tsv", "amazon_reviews_us_Office_Products_v1_00.tsv", "amazon_reviews_us_Outdoors_v1_00.tsv", "amazon_reviews_us_PC_v1_00.tsv", "amazon_reviews_us_Personal_Care_Appliances_v1_00.tsv", "amazon_reviews_us_Pet_Products_v1_00.tsv", "amazon_reviews_us_Shoes_v1_00.tsv", "amazon_reviews_us_Software_v1_00.tsv", "amazon_reviews_us_Sports_v1_00.tsv", "amazon_reviews_us_Tools_v1_00.tsv", "amazon_reviews_us_Toys_v1_00.tsv", "amazon_reviews_us_Video_DVD_v1_00.tsv", "amazon_reviews_us_Video_Games_v1_00.tsv", "amazon_reviews_us_Video_v1_00.tsv", "amazon_reviews_us_Watches_v1_00.tsv", "amazon_reviews_us_Wireless_v1_00.tsv"]
```

4. Created download_files.py (nano download_files.py) and inserted the following:

```
import os
import json

bucket_path = "s3://amazon-reviews-ea/landing"
file = open("kaggle_filenames.json")
filenames = json.load(file)

for i, filename in enumerate(filenames):
        os.system(f"kaggle datasets download -d cynthiarempel/amazon-us-customer-reviews-dataset -f {filename}")
        print("Unzipping {filename}.zip . . .")
        os.system(f"unzip {filename}.zip")
        print("Uploading to S3 Bucket . . .")
        os.system(f"aws s3 cp {filename} {bucket_path}/{filename}")
        print(f"Removing {filename} from EC2 . . .")
        if os.path.exists(filename):
                os.remove(filename)
        if os.path.exists(f"{filename}.zip"):
                os.remove(f"{filename}.zip")
        print(f"Completed {i+1}/{len(filenames)}!\n")
```

5. Execute download_files.py (python3 download_files.py)
6. Viewed files in AWS S3 Bucket by executing the following command:
```
aws s3 ls s3://amazon-reviews-ea/landing/ --human-readable --summarize
```

## Exploratory Data Analysis
```
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
    df = pd.read_csv(path, usecols=cols, sep='\t', error_bad_lines=False, low_memory=False)

    # Encoded 'vine' and 'verified_purchase' columns for analysis
    df['vine'] = df['vine'].apply(lambda x: 1 if x == 'Y' else 0 if x == "N" else None)
    df['verified_purchase'] = df['verified_purchase'].apply(lambda x: 1 if x == 'Y' else 0)
    
    # Counting number of words in the 'review_headline' and 'review_body' columns
    df['review_headline_word_count'] = df['review_headline'].apply(lambda headline: len(str(headline).split()))
    df['review_body_word_count'] = df['review_body'].apply(lambda body: len(str(body).split()))

    # Selecting columns I am interested in analyzing
    num_cols = ['star_rating', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase', 'review_headline_word_count', 'review_body_word_count']
    text_cols = ['review_headline', 'review_body']
    date_col = 'review_date'
    
    # Changed 'review_date' type to datetime 'star_rating' type to be float64
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    df['star_rating'] = pd.to_numeric(df['star_rating'], errors='coerce')
    
    # Printing output for analysis
    print(f"List of Variables: {list(df.columns)}")
    print(f"Number of Observations: {df.shape[0]}")
    print(f"Number of Customers: {len(df['customer_id'].unique())}")
    print(f"Number of Products: {len(df['product_id'].unique())}")
    print(f"Number of Duplicates: {df.duplicated(subset='review_id', keep='first').sum()}")
    print(f"Number of Missing/Null Values: {df.isna().sum().sum()}")
    print(f"Number of Missing/Null Values per Affected Column:")
    display(pd.DataFrame(df[df.columns[df.isnull().any()].to_list()].isna().sum(), columns=['count']).T)
    print(f"Min Review Date: {min(df[date_col])}")
    print(f"Max Review Date: {max(df[date_col])}")
    print(f"Min/Max/Avg/Stdev for Numerical Variables:")
    display(df[num_cols].agg(['min', 'max', 'mean', 'std']).round(decimals=3))
print("="*120)
```

## Feature Engineering and Modeling
1. Import the necessary libraries for this milestone
```
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
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, HashingTF, IDF, Tokenizer, RegexTokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import *
from pyspark.ml.tuning import *
import numpy as np
from textblob import TextBlob
```

2. Prepared environment variables and variables save paths and filenames
```
access_key = "---INSERT ACCESS KEY---"
secret_key = "---INSERT SECRET KEY---"
os.environ['AWS_ACCESS_KEY_ID'] = access_key
os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
aws_region = "us-east-2"

sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", access_key)
sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", secret_key)
sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3." + aws_region + ".amazonaws.com")

bucket_path = "s3://amazon-reviews-ea/"
bucket_name = 'amazon-reviews-ea'

s3_client = boto3.client("s3")
response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="landing/")
objects = response.get("Contents")

filenames = [obj['Key'] for obj in objects][1:]
```

3. Cleaned the data in the landing directory from the Amazon S3 bucket using PySpark and DataBricks and saved the clean data in a parquet file in the raw directory 
```
@udf
def ascii_only(text):
    return text.encode('ascii', 'ignore').decode('ascii') if text else None

for i, filename in enumerate(filenames):
    input_file_path = bucket_path + filename
    output_file_path = f"s3://amazon-reviews-ea/raw/cleaned_{filename[8:]}"[:-3] + "parquet"
    print(f"{i+1}/{len(filenames)} {input_file_path} ---> {output_file_path}")

    sdf = spark.read.csv(input_file_path, sep='\t', header=True, inferSchema=True)

    # Select columns
    cols = ['marketplace', 'product_title', 'product_category', 'star_rating', 'helpful_votes', 'total_votes', 'verified_purchase', 'review_headline', 'review_body', 'review_date']
    sdf = sdf.select(cols)

    # Clean text-based columns
    sdf = sdf.withColumn("review_headline", ascii_only('review_headline'))
    sdf = sdf.withColumn("review_body", ascii_only('review_body'))

    # Remove null values
    sdf = sdf.na.drop(subset=["star_rating", "review_body"])

    # Drop duplicates
    sdf = sdf.dropDuplicates()

    # Save the file in Amazon S3
    sdf.write.parquet(output_file_path)
```

4. Aggregated all of the parquet files into a singular PySpark DataFrame (main_sdf)
```
response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="raw/")
objects = response.get("Contents")
filenames = list(set([obj['Key'][:obj['Key'].find('.')+8] for obj in objects][1:]))

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

```

5. Created and applied the model pipeline
```
indexer_1 = StringIndexer(inputCol="product_category", outputCol="product_category_index")
regexTokenizer_1 = RegexTokenizer(inputCol="review_body", outputCol="review_body_tokens", pattern="\\w+", gaps=False)
regexTokenizer_2 = RegexTokenizer(inputCol="review_headline", outputCol="review_headline_tokens", pattern="\\w+", gaps=False)
regexTokenizer_3 = RegexTokenizer(inputCol="product_title", outputCol="product_title_tokens", pattern="\\w+", gaps=False)
pipeline = Pipeline(stages=[indexer_1, regexTokenizer_1, regexTokenizer_2, regexTokenizer_3])
main_sdf = pipeline.fit(main_sdf).transform(main_sdf)
```

6. Conducted more feature engineering with additional encoding, sentiment analysis, and feature extraction from text-based features
```
@udf
def sentiment_analysis(text):
    sentiment = TextBlob(text).sentiment.polarity
    return sentiment

main_sdf = main_sdf.withColumn("verified_purchase", (col("verified_purchase") == "Y").cast("int"))
main_sdf = main_sdf.withColumn("sentiment_score", sentiment_analysis(col("review_body")))
main_sdf = main_sdf.withColumn('review_headline_word_count', get_word_count(col("review_headline")))
main_sdf = main_sdf.withColumn('review_body_word_count', get_word_count(col("review_body")))
main_sdf = main_sdf.drop(*["review_headline", "review_body", "product_title","product_category"])
```

7. Casted variables to appropriate data types for the VectorAssembler
```
main_sdf = main_sdf.withColumn("review_headline_word_count", main_sdf.review_headline_word_count.cast('double'))
main_sdf = main_sdf.withColumn("review_body_word_count", main_sdf.review_body_word_count.cast('double'))
main_sdf = main_sdf.withColumn("sentiment_score", main_sdf.sentiment_score.cast('double'))
main_sdf = main_sdf.withColumn("star_rating", main_sdf.star_rating.cast('double'))
```

8. Used hashingTF and IDF to encode text-based tokens
```
hashingTF = HashingTF(numFeatures=4096, inputCol="review_body_tokens", outputCol="review_body_hashed")
main_sdf = hashingTF.transform(main_sdf)
idfModel = IDF(inputCol='review_body_hashed', outputCol="review_body_features", minDocFreq=1).fit(main_sdf)
main_sdf = idfModel.transform(main_sdf)
hashingTF = HashingTF(numFeatures=4096, inputCol="review_headline_tokens", outputCol="review_headline_hashed")
main_sdf = hashingTF.transform(main_sdf)
idfModel = IDF(inputCol='review_headline_hashed', outputCol="review_headline_features", minDocFreq=1).fit(main_sdf)
main_sdf = idfModel.transform(main_sdf)
hashingTF = HashingTF(numFeatures=4096, inputCol="product_title_tokens", outputCol="product_title_hashed")
main_sdf = hashingTF.transform(main_sdf)
idfModel = IDF(inputCol='product_title_hashed', outputCol="product_title_features", minDocFreq=1).fit(main_sdf)
main_sdf = idfModel.transform(main_sdf)
main_sdf = main_sdf.drop(*['review_body_tokens', 'review_headline_tokens', 'product_title_tokens'])
```

9. Created and evaluated the model, and uploaded the data and model to Amazon S3
```
input_cols = [el for el in main_sdf.columns if el != 'star_rating']
assembler = VectorAssembler(inputCols=input_cols, outputCol='features')
main_sdf = assembler.transform(main_sdf)
main_sdf = main_sdf.withColumn('label', when(col('star_rating') > 3, 1.0).otherwise(0.0))
sdf = main_sdf.select(['features', 'label'])
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
cm = test_results.groupby('label').pivot('prediction').count().fillna(0).collect()
def calculate_recall_precision(cm):
  tn = cm[0][1] # True Negative
  fp = cm[0][2] # False Positive
  fn = cm[1][1] # False Negative
  tp = cm[1][2] # True Positive
  precision = tp / ( tp + fp )
  recall = tp / ( tp + fn )
  accuracy = ( tp + tn ) / ( tp + tn + fp + fn )
  f1_score = 2 * ( ( precision * recall ) / ( precision + recall ) )
  return accuracy, precision, recall, f1_score
print(calculate_recall_precision(cm))
```

10. Use the Cross-Validator to find the best model
```
# Create a BinaryClassificationEvaluator to evaluate how well the model works
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")

# Create the parameter grid (empty for now)
grid = ParamGridBuilder().build()

# Create the CrossValidator
cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, numFolds=3)

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
print('Number of models to be tested: ', len(grid))

# Create the CrossValidator using the new hyperparameter grid
cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)

# Call cv.fit() to create models with all of the combinations of parameters in the grid
all_models = cv.fit(trainingData)
print("Average Metrics for Each model: ", all_models.avgMetrics)

model = all_models.bestModel
print("Area under ROC curve:", model.summary.areaUnderROC)  # ~0.87
```

**Summarization:** The first step was to read in all of the data in the "landing" directory in Amazon S3, conduct standard data cleaning processes, then write it back to Amazon S3 under the "raw" directory. Then, I read back all of the parquet files from the "raw" directory in Amazon S3 and conducted standard feature engineering techniques, such as encoding, feature extraction, and IDF. After the data was trusted enough for creating a Logistic Regression model, I wrote the data back into Amazon S3 under the "trusted" directory. I then created and tested the model after splitting the data into training and testing datasets. Finally, I used the CrossValidator object and a ParamGrid to figure out my best model. The results that are presented later on are based on the best model, which had a regParam of 0.6 and an elasticNetParam of 1. There were many challenges when completing this milestone. For example, there were issues regarding computing time (ex. model training). There were issues regarding costs of storage with regards to Amazon S3. Other issues were overcome easily, such as dealing with data types and casting, the PySpark library and documentation, and encoding features appropriately. Due to issues with computing power and limited resources, I was only able to create a model using a small subset of the original dataset, **which concluded with an accuracy score of approximately 82%, a precision score of approximately 93%, a recall score of approximately 83%, a F1 score of approximately 88%, and finally an AUC of 0.87.** Although these metrics are quite "high" for the model, it is important to continuously test the model to make sure that the model is not biased or misleading in any way. It is definitely a good start and will continue to improve as I introduce the rest of the data stored in the "raw" directory to the Logistic Regression model.


## Data Visualizing

#### Visualization #1: Review Body Word Count vs. Star Rating
```
df = main_sdf.select(['review_body_word_count', 'star_rating']).toPandas()
fig = plt.figure(facecolor='white')
plt.xlabel("Star Rating (1-5)")
plt.ylabel("Number of Words in Review Body")
plt.title("Number of Review Body Words vs. Star Rating")
plt.xticks([1,2,3,4,5])
fig.tight_layout()
plt.scatter(df['star_rating'], df['review_body_word_count'], alpha=0.5, s=100)
plt.show()
```

#### Visualization #2: Average Review Body Word Count vs. Star Rating
```
df = main_sdf.select(['review_body_word_count', 'star_rating']).groupby('star_rating').agg(avg('review_body_word_count').alias('review_body_word_count')).toPandas().sort_values(by='star_rating')
fig = plt.figure(facecolor='white')
plt.xlabel("Star Rating (1-5)")
plt.ylabel("Number of Words in Review Body")
plt.title("Average Number of Review Body Words vs. Star Rating")
plt.xticks([1,2,3,4,5])
fig.tight_layout()
plt.plot(df['star_rating'], df['review_body_word_count'], alpha=0.5)
plt.show()
```

#### Visualization #3: Sentiment Score vs. Star Rating
```
df = main_sdf.select(['sentiment_score', 'star_rating']).groupby('star_rating').agg(avg('sentiment_score').alias('sentiment_score')).toPandas().sort_values(by='star_rating') 
fig = plt.figure(facecolor='white')
plt.xlabel("Star Rating (1-5)")
plt.ylabel("Review Body Sentiment Score [-1, 1]")
plt.title("Review Body Sentiment Score vs. Star Rating")
plt.xticks([1,2,3,4,5])
fig.tight_layout()
plt.plot(df['star_rating'], df['sentiment_score'], alpha=0.5)
plt.show()
```

#### Visualization #4: Correlation Matrix
```
vector_column = "correlation_features"
numeric_columns = ['helpful_votes', 'total_votes', 'verified_purchase', 'sentiment_score', 'review_headline_word_count', 'review_body_word_count']
assembler = VectorAssembler(inputCols=numeric_columns, outputCol=vector_column)
sdf_vector = assembler.transform(main_sdf).select(vector_column)
matrix = Correlation.corr(sdf_vector, vector_column).collect()[0][0]
correlation_matrix = matrix.toArray().tolist()
correlation_matrix_df = pd.DataFrame(data=correlation_matrix, columns=numeric_columns,
index=numeric_columns)
sns.set_style("white")
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix_df,
xticklabels=correlation_matrix_df.columns.values,
yticklabels=correlation_matrix_df.columns.values, cmap="Greens", annot=True)
plt.savefig("correlation_matrix.png")
```

#### Visualization #5: Confusion Matrix
```
test_results.groupby('label').pivot('prediction').count().sort('label').show()
```

#### Visualization #6: ROC Curve
```
plt.figure(figsize=(6,6))
plt.plot([0, 1], [0, 1], 'r--')
x = model.summary.roc.select('FPR').collect()
y = model.summary.roc.select('TPR').collect()
plt.scatter(x, y)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")
```

## Summary and Conclusions
- The objective of this project was to create a machine learning pipeline that can predict if the star rating of an Amazon product is greater than 3 based on its various features using popular cloud technologies such as Amazon EC2, Amazon S3, and DataBricks. The first step of this project was to acquire the data. This data was acquired from Kaggle, an online community where users upload a large plethora of datasets for data analysis and machine learning use cases. This was done via an Amazon EC2 instance, where all of the selected files were downloaded and stored in my own personal Amazon S3 bucket. Now that all of the data is safely stored, it is now ready for analysis.
- The second step of this project was to conduct exploratory data analysis that was stored in the Amazon S3 bucket from the previous step. Within the same Amazon EC2 instance, I loaded each file into a Pandas DataFrame, filtered selected columns, and analyzed them. This includes looking at the shape of the DataFrame, the number products, duplicates, null values, minimum and maximum values for date columns, and general statistical information for the numerical variables. From this, we can identify ways we should clean the data (removing nulls, outliers, etc.) and handle features appropriately.
- The third step of this project was to conduct feature engineering and modeling, which was done in DataBricks, a cloud-based data engineering tool used for processing and transforming large amounts of data. I removed null values and duplicated data. I cleaned the text-based columns so that it only contained valid text. After that, I've then aggregated all of the data into one Spark DataFrame. Then, I created a model pipeline that vectorizes all non-numerical data, and tokenized all of the long-text data into tokens. After, I calculated the numerical sentiment of the body of each review and casted selected features to desired data types. Once I've completed that, I hashed all of the tokens from appropriate columns and randomly split the data into training and testing sets, training to train the model and testing to test the model's performance.
- The final step was to train a logistic regression model that predicts whether a review is above a 3 star rating. In pursuit of finding the best model, I've used a CrossValidator and a ParamGrid to train 12 models, keeping the best model for testing against the testing data to collect metrics on its performance. The metrics of the best model contained an accuracy score of approximately 82%, a precision score of approximately 93%, a recall score of approximately 83%, a F1 score of approximately 88%, and finally an AUC of 0.87. To effectively showcase the performance of the model and the nature of the data, I created 5 data visualizations to show not only the relationship between the most important features via bar charts, line charts, and a correlation matrix, but also the accuracy of the model via a confusion matrix and a ROC curve.
- In terms of the main conclusions I was able to draw from the data, the overall integrity of the data was high. It was structured very well, and required minimal amounts of cleaning. There were some instances of multicollinearity, but it did not interfere much with appropriately training the model. It was also very clear that the star ratings and the sentiment scores were directly correlated, which meant that if a review were to be given a higher rating on the 1-5 star scale, the sentiment of the review body would most likely be positive. 
- Overall, this project has taught me a great amount, especially with regards to how to utilize cloud technologies to process and transform big data. I learned how to not only write cleaner code, but to become more cautious on what kind of code I am running. With years of experience of running code locally, I did not really pay attention to how resource-intensive my code can be. With cloud computing and the pay-as-you-go pricing structure, this can lead to a very expensive bill. I also learned how to create a machine learning pipeline, utilizing very well-known industry standard technologies, such as Amazon EC2, Amazon S3, and DataBricks.
