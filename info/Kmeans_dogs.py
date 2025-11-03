#Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv('dogsofzurich2015.csv')
print(df)

# Data Cleaning
print(df.describe())





