# ================================================================================================== #
# Visualization #1: Review Body Word Count vs. Star Rating
# ================================================================================================== #
df = main_sdf.select(["review_body_word_count", "star_rating"]).toPandas()
fig = plt.figure(facecolor="white")
plt.xlabel("Star Rating (1-5)")
plt.ylabel("Number of Words in Review Body")
plt.title("Number of Review Body Words vs. Star Rating")
plt.xticks([1, 2, 3, 4, 5])
fig.tight_layout()
plt.scatter(df["star_rating"], df["review_body_word_count"], alpha=0.5, s=100)
plt.show()


# ================================================================================================== #
# Visualization #2: Average Review Body Word Count vs. Star Rating
# ================================================================================================== #
df = (
    main_sdf.select(["review_body_word_count", "star_rating"])
    .groupby("star_rating")
    .agg(avg("review_body_word_count").alias("review_body_word_count"))
    .toPandas()
    .sort_values(by="star_rating")
)
fig = plt.figure(facecolor="white")
plt.xlabel("Star Rating (1-5)")
plt.ylabel("Number of Words in Review Body")
plt.title("Average Number of Review Body Words vs. Star Rating")
plt.xticks([1, 2, 3, 4, 5])
fig.tight_layout()
plt.plot(df["star_rating"], df["review_body_word_count"], alpha=0.5)
plt.show()


# ================================================================================================== #
# Visualization #3: Sentiment Score vs. Star Rating
# ================================================================================================== #
df = (
    main_sdf.select(["sentiment_score", "star_rating"])
    .groupby("star_rating")
    .agg(avg("sentiment_score").alias("sentiment_score"))
    .toPandas()
    .sort_values(by="star_rating")
)
fig = plt.figure(facecolor="white")
plt.xlabel("Star Rating (1-5)")
plt.ylabel("Review Body Sentiment Score [-1, 1]")
plt.title("Review Body Sentiment Score vs. Star Rating")
plt.xticks([1, 2, 3, 4, 5])
fig.tight_layout()
plt.plot(df["star_rating"], df["sentiment_score"], alpha=0.5)
plt.show()


# ================================================================================================== #
# Visualization #4: Correlation Matrix
# ================================================================================================== #
vector_column = "correlation_features"
numeric_columns = [
    "helpful_votes",
    "total_votes",
    "verified_purchase",
    "sentiment_score",
    "review_headline_word_count",
    "review_body_word_count",
]
assembler = VectorAssembler(inputCols=numeric_columns, outputCol=vector_column)
sdf_vector = assembler.transform(main_sdf).select(vector_column)
matrix = Correlation.corr(sdf_vector, vector_column).collect()[0][0]
correlation_matrix = matrix.toArray().tolist()
correlation_matrix_df = pd.DataFrame(
    data=correlation_matrix, columns=numeric_columns, index=numeric_columns
)
sns.set_style("white")
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix_df,
    xticklabels=correlation_matrix_df.columns.values,
    yticklabels=correlation_matrix_df.columns.values,
    cmap="Greens",
    annot=True,
)
plt.savefig("correlation_matrix.png")


# ================================================================================================== #
# Visualization #5: Confusion Matrix
# ================================================================================================== #
test_results.groupby("label").pivot("prediction").count().sort("label").show()


# ================================================================================================== #
# Visualization #6: ROC Curve
# ================================================================================================== #
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], "r--")
x = model.summary.roc.select("FPR").collect()
y = model.summary.roc.select("TPR").collect()
plt.scatter(x, y)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
