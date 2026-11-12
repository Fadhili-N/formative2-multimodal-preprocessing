import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Load customer transactions
transactions = pd.read_csv(Path(__file__).with_name("customer_transactions.csv"))

# Load customer social profiles
social_profiles = pd.read_csv(Path(__file__).with_name("customer_social_profiles.csv"))

# Display first five rows
print("TRANSACTIONS")
print(transactions.head())

print("\nSOCIAL PROFILES")
print(social_profiles.head())

# Check dataset dimensions
print("Transactions shape:", transactions.shape)
print("Social profiles shape:", social_profiles.shape)

# Check column names and data types
print("\nTRANSACTIONS INFO")
transactions.info()

print("\nSOCIAL PROFILES INFO")
social_profiles.info()

# Statistical summary
print("\nTRANSACTIONS SUMMARY")
print(transactions.describe())

print("\nSOCIAL PROFILES SUMMARY")
print(social_profiles.describe())

print("Missing values in transactions:")
print(transactions.isnull().sum())

print("\nMissing values in social profiles:")
print(social_profiles.isnull().sum())

print("Transaction duplicates:", transactions.duplicated().sum())
print("Social duplicates:", social_profiles.duplicated().sum())

# Remove exact duplicate rows
transactions = transactions.drop_duplicates()
social_profiles = social_profiles.drop_duplicates()

# Convert purchase date to datetime
transactions["purchase_date"] = pd.to_datetime(
    transactions["purchase_date"],
    errors="coerce"
)

# Convert purchase amount to numeric
transactions["purchase_amount"] = pd.to_numeric(
    transactions["purchase_amount"],
    errors="coerce"
)

# Convert customer rating to numeric
transactions["customer_rating"] = pd.to_numeric(
    transactions["customer_rating"],
    errors="coerce"
)

# Convert social scores to numeric
social_profiles["engagement_score"] = pd.to_numeric(
    social_profiles["engagement_score"],
    errors="coerce"
)

social_profiles["purchase_interest_score"] = pd.to_numeric(
    social_profiles["purchase_interest_score"],
    errors="coerce"
)

# Fill missing purchase amounts with median
transactions["purchase_amount"] = transactions[
    "purchase_amount"
].fillna(transactions["purchase_amount"].median())

# Fill missing ratings with median
transactions["customer_rating"] = transactions[
    "customer_rating"
].fillna(transactions["customer_rating"].median())

print(transactions.isnull().sum())

print(social_profiles.isnull().sum())

print(transactions.dtypes)

# Convert transaction customer IDs to numeric
transactions["customer_id"] = pd.to_numeric(
    transactions["customer_id_legacy"],
    errors="coerce"
)

# Remove the A prefix from social customer IDs
social_profiles["customer_id"] = pd.to_numeric(
    social_profiles["customer_id_new"].str.replace("A", "", regex=False),
    errors="coerce"
)

print("Transaction IDs:")
print(transactions["customer_id"].head())

print("\nSocial profile IDs:")
print(social_profiles["customer_id"].head())

transaction_customers = set(transactions["customer_id"].dropna())

social_customers = set(social_profiles["customer_id"].dropna())

# Customers appearing in both datasets
common_customers = transaction_customers.intersection(
    social_customers
)

print("Customers in transactions:", len(transaction_customers))

print("Customers in social profiles:", len(social_customers))

print("Matching customers:", len(common_customers))

# Create a function to get the most common value
def most_common_value(series):
    return series.mode().iloc[0] if not series.mode().empty else "Unknown"


# Aggregate social profiles by customer
social_aggregated = social_profiles.groupby("customer_id").agg({

    "social_media_platform": ["nunique", most_common_value],

    "engagement_score": "mean",

    "purchase_interest_score": "mean",

    "review_sentiment": most_common_value

}).reset_index()

social_aggregated.columns = [
    "_".join(col).strip("_") if isinstance(col, tuple) else col
    for col in social_aggregated.columns
]

print(social_aggregated.head())

merged_dataset = pd.merge(

    transactions,

    social_aggregated,

    on="customer_id",

    how="left"

)

# Count the number of social media platforms per customer

platform_count = social_profiles.groupby("customer_id")[
    "social_media_platform"
].nunique()

# Add platform count to aggregated social profiles

social_aggregated["social_platform_count"] = (
    social_aggregated["customer_id"].map(platform_count)
)

# Check dimensions
print("Original transactions:", transactions.shape)

print("Merged dataset:", merged_dataset.shape)


# Check duplicate transaction IDs
print(
    "Duplicate transactions:",
    merged_dataset["transaction_id"].duplicated().sum()
)


# Check missing values
print("\nMissing values:")
print(merged_dataset.isnull().sum())


# Display merged dataset
print("\nMerged dataset:")
print(merged_dataset.head())

merged_dataset.to_csv("merged_dataset.csv", index=False)

print("Merged dataset successfully saved!")

def classify_engagement(score):

    if pd.isna(score):
        return "Unknown"

    elif score >= 70:
        return "High"

    elif score >= 40:
        return "Medium"

    else:
        return "Low"


social_aggregated["engagement_level"] = (
    social_aggregated["engagement_score_mean"].apply(classify_engagement)
)

# Sort transactions by customer and date

transactions = transactions.sort_values(
    ["customer_id", "purchase_date"]
).copy()


# Number of previous purchases

transactions["prior_purchase_count"] = (
    transactions.groupby("customer_id").cumcount()
)


# Average amount spent on previous purchases

transactions["prior_avg_purchase_amount"] = (
    transactions.groupby("customer_id")["purchase_amount"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)


# Average rating from previous purchases

transactions["prior_avg_customer_rating"] = (
    transactions.groupby("customer_id")["customer_rating"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)

# Merge transaction features with social features

merged_dataset = pd.merge(
    transactions,
    social_aggregated,
    on="customer_id",
    how="left"
)

print(merged_dataset.columns.tolist())

merged_dataset.to_csv(
    "merged_dataset.csv",
    index=False
)

print("Feature engineering completed!")


# Features used for prediction
numeric_features = [
    "engagement_score",
    "purchase_interest_score",
    "social_platform_count",
    "prior_purchase_count",
    "prior_avg_purchase_amount",
    "prior_avg_customer_rating"
]

categorical_features = [
    "social_media_platform",
    "review_sentiment"
]

feature_columns = numeric_features + categorical_features

X = merged_dataset[feature_columns]

# Target variable
y = merged_dataset["product_category"]

# Remove rows without a target
model_data = merged_dataset.dropna(
    subset=["product_category"]
).copy()

X = model_data[feature_columns]
y = model_data["product_category"].astype(str)

groups = model_data["customer_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessing", preprocessor),

    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2
    ))
])

model.fit(X_train, y_train)

print("Model training completed!")

predictions = model.predict(X_test)

print("Actual products:")
print(y_test.head(10).tolist())

print("\nPredicted products:")
print(predictions[:10])

probabilities = model.predict_proba(X_test)

print(probabilities[:5])

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
print("Model Accuracy Percentage:", accuracy * 100, "%")

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

print("Precision:", precision)
print("Recall:", recall)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

print("F1 Score:", f1)

probabilities = model.predict_proba(X_test)

loss = log_loss(
    y_test,
    probabilities,
    labels=model.classes_
)

print("Log Loss:", loss)

report = classification_report(
    y_test,
    predictions,
    zero_division=0
)

print(report)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

plt.figure(figsize=(9, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Product")
plt.ylabel("Actual Product")
plt.title("Product Recommendation Confusion Matrix")

plt.tight_layout()
plt.show()

evaluation_results = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "Log Loss": loss
}

print("\nMODEL EVALUATION RESULTS")

for metric, value in evaluation_results.items():
    print(f"{metric}: {value:.4f}")

with open("evaluation_results.txt", "w") as file:

    file.write("PRODUCT RECOMMENDATION MODEL EVALUATION\n")

    for metric, value in evaluation_results.items():

        file.write(f"{metric}: {value:.4f}\n")

    file.write("\nCLASSIFICATION REPORT\n")

    file.write(report)
    
# Save the trained model

joblib.dump(model, "product_recommendation_model.joblib")

print("Model successfully saved!")

# Load the saved model

loaded_model = joblib.load(
    "product_recommendation_model.joblib"
)

print("Model successfully loaded!")

# Select one customer from the test dataset

new_customer = X_test.iloc[[0]]

# Predict the product category

prediction = loaded_model.predict(new_customer)

print("Recommended product:", prediction[0])

probabilities = loaded_model.predict_proba(new_customer)

# Get product categories
categories = loaded_model.classes_

# Display probabilities
for category, probability in zip(categories, probabilities[0]):

    print(f"{category}: {probability:.2%}")

best_index = probabilities[0].argmax()

recommended_product = categories[best_index]

print("Best recommendation:", recommended_product)

if os.path.exists("product_recommendation_model.joblib"):

    print("Model file exists!")

    print(
        "File size:",
        os.path.getsize("product_recommendation_model.joblib"),
        "bytes"
    )

else:

    print("Model file was not found.")