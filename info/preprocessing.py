import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset using a relative path
# 
base_dir = os.path.dirname(__file__)  # folder where this .py file is
csv_path = os.path.join(base_dir, "dogsofzurich2015.csv")

df = pd.read_csv(csv_path, encoding="latin1")

# Inspect data
print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== DATASET SHAPE =====")
print(df.shape)

#Handle birth year and compute age 
if "birth_year" in df.columns:
    df["birth_year"] = pd.to_numeric(df["birth_year"], errors="coerce")
    df["age"] = 2015 - df["birth_year"]
    print("\n===== AGE SUMMARY =====")
    print(df["age"].describe())

# Check for missing values 
print("\n===== MISSING VALUES PER COLUMN =====")
print(df.isna().sum())

# Show top values for key columns 
for col in ["breed", "gender", "district"]:
    if col in df.columns:
        print(f"\n===== TOP {col.upper()} VALUES =====")
        print(df[col].value_counts().head(10))

# Plot age distribution
if "age" in df.columns:
    df["age"].hist(bins=30, color="skyblue", edgecolor="black")
    plt.title("Dog Age Distribution (Zurich 2015)")
    plt.xlabel("Age (years)")
    plt.ylabel("Number of Dogs")
    plt.tight_layout()
    plt.show()

# cleaning
print("\n===== STARTING CLEANING =====")

df.columns = [c.strip().upper() for c in df.columns]

# Drop columns that are completely empty (like BREED2_MIX)
df = df.dropna(axis=1, how="all")

# Fill in missing numeric values
df["CITY_ID"] = df["CITY_ID"].fillna(df["CITY_ID"].median())
df["CITY_DISTRICT"] = df["CITY_DISTRICT"].fillna(df["CITY_DISTRICT"].median())

# Fill missing categorical values
for col in ["BREED1", "RACE_TYPE", "DOGS_COLOR", "GENDER", "DOGS_GENDER"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# Create numeric age column
df["DOGS_YEAR_OF_BIRTH"] = pd.to_numeric(df["DOGS_YEAR_OF_BIRTH"], errors="coerce")
df["AGE_YEARS"] = 2015 - df["DOGS_YEAR_OF_BIRTH"]

print("\nAge column summary:")
print(df["AGE_YEARS"].describe())

# Simplify breeds
top_breeds = df["BREED1"].value_counts().nlargest(30).index
df["BREED_REDUCED"] = df["BREED1"].where(df["BREED1"].isin(top_breeds), "Other")

print("\nTop 10 breeds after simplification:")
print(df["BREED_REDUCED"].value_counts().head(10))

# Gender encoding
df["DOGS_GENDER"] = df["DOGS_GENDER"].map({"Rüde": "Male", "Hündin": "Female"}).fillna("Unknown")

# One-hot encode key columns
df_encoded = pd.get_dummies(df, columns=["BREED_REDUCED", "DOGS_GENDER"], drop_first=True)

# Scale numeric features
numeric_cols = ["AGE_YEARS", "CITY_ID", "CITY_DISTRICT"]
scaler = StandardScaler()
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

#Save cleaned dataset 
cleaned_path = os.path.join(base_dir, "dogs_cleaned.csv")
df_encoded.to_csv(cleaned_path, index=False)

print(f"\n CLEANED DATA SAVED to:\n{cleaned_path}")
print("Shape after encoding:", df_encoded.shape)
print("Sample preview:")
print(df_encoded.head())
