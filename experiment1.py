
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Titanic.csv")

print("\n========================================")
print("DATASET LOADED SUCCESSFULLY")
print("========================================")

print("\nFirst 5 Records:")
print(df.head())

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

df.info()

print("\nStatistical Description:")
print(df.describe())

print("\nDataset Shape:")
print(df.shape)

print("\n========================================")
print("MISSING VALUES BEFORE CLEANING")
print("========================================")

print(df.isnull().sum())

if "Age" in df.columns:
    df["Age"] = df["Age"].fillna(df["Age"].median())
if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
if "Fare" in df.columns:
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)


print("\n========================================")
print("MISSING VALUES AFTER CLEANING")
print("========================================")

print(df.isnull().sum())

print("\n========================================")
print("DUPLICATE RECORDS")
print("========================================")

print("Duplicates before removal:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("Duplicates after removal:", df.duplicated().sum())
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    df[column] = df[column].astype(str).str.strip()

print("\nCategorical Columns:")
print(list(categorical_columns))

columns_to_encode = []

if "Sex" in df.columns:
    columns_to_encode.append("Sex")

if "Embarked" in df.columns:
    columns_to_encode.append("Embarked")

if len(columns_to_encode) > 0:
    df = pd.get_dummies(
        df,
        columns=columns_to_encode,
        drop_first=True,
        dtype=int
    )

print("\n========================================")
print("DATA AFTER ENCODING")
print("========================================")

print(df.head())
if "Age" in df.columns:

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["Age"])
    plt.title("Box Plot - Age")
    plt.xlabel("Age")
    plt.show()


if "Fare" in df.columns:

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["Fare"])
    plt.title("Box Plot - Fare")
    plt.xlabel("Fare")
    plt.show()

def find_iqr_limits(data, column):

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_limit) |
        (data[column] > upper_limit)
    ]

    print("\nColumn:", column)
    print("Q1 =", Q1)
    print("Q3 =", Q3)
    print("IQR =", IQR)
    print("Lower Limit =", lower_limit)
    print("Upper Limit =", upper_limit)
    print("Number of Outliers =", len(outliers))

    return lower_limit, upper_limit


if "Age" in df.columns:
    age_lower, age_upper = find_iqr_limits(df, "Age")

if "Fare" in df.columns:
    fare_lower, fare_upper = find_iqr_limits(df, "Fare")

if "Age" in df.columns:

    df["Age"] = np.clip(
        df["Age"],
        age_lower,
        age_upper
    )


if "Fare" in df.columns:

    df["Fare"] = np.clip(
        df["Fare"],
        fare_lower,
        fare_upper
    )

print("\nOutliers treated using IQR capping.")

if "SibSp" in df.columns and "Parch" in df.columns:

    df["FamilySize"] = (
        df["SibSp"] +
        df["Parch"] +
        1
    )


if "FamilySize" in df.columns:

    df["IsAlone"] = np.where(
        df["FamilySize"] == 1,
        1,
        0
    )

def create_age_group(age):

    if age < 13:
        return "Child"

    elif age < 20:
        return "Teenager"

    elif age < 60:
        return "Adult"

    else:
        return "Senior"


if "Age" in df.columns:

    df["AgeGroup"] = df["Age"].apply(create_age_group)


print("\n========================================")
print("FEATURE ENGINEERING")
print("========================================")

columns_to_display = []

for column in ["Age", "FamilySize", "IsAlone", "AgeGroup"]:

    if column in df.columns:
        columns_to_display.append(column)

print(df[columns_to_display].head())

if "AgeGroup" in df.columns:

    df = pd.get_dummies(
        df,
        columns=["AgeGroup"],
        drop_first=True,
        dtype=int
    )


numerical_columns = df.select_dtypes(
    include=np.number
).columns

print("\n========================================")
print("NUMERICAL COLUMNS")
print("========================================")

print(list(numerical_columns))

scaler = MinMaxScaler()

df[numerical_columns] = scaler.fit_transform(
    df[numerical_columns]
)


print("\nNormalization completed successfully.")

print("\n========================================")
print("FINAL DATASET")
print("========================================")

print(df.head(10))

print("\nFinal Shape:")
print(df.shape)

print("\nFinal Missing Values:")
print(df.isnull().sum())

print("\nFinal Duplicate Count:")
print(df.duplicated().sum())

output_file = "Titanic_Cleaned_Preprocessed.csv"

df.to_csv(
    output_file,
    index=False
)

print("\n========================================")
print("PROCESS COMPLETED")
print("========================================")

print("Cleaned dataset saved as:")
print(output_file)