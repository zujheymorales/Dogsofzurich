#Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "dogs_cleaned.csv")
df = pd.read_csv(csv_path)

# Use data to determine clusters
print(df.describe())

#Select only numerical and scale
numeric_df = df.select_dtypes(include=['number', 'bool'])
scaler = StandardScaler()
scaled_arr = scaler.fit_transform(numeric_df.values)
scaled_df = pd.DataFrame(scaled_arr, columns=numeric_df.columns, index=numeric_df.index)
print("Data scaled successfully")
#Plot to detemine how many clusters would be optimal
Kmeans_kwargs = {
    "init": "random",
    "n_init": 5,
    "random_state": 3,
}
sse = []
for k in range(1, 5):
    kmeans = KMeans(n_clusters=k, **Kmeans_kwargs)
    kmeans.fit(scaled_df)
    sse.append(kmeans.inertia_)

#visualize the results
plt.plot(range(1, 5), sse, marker ="o")
plt.xticks(range(1, 5))
plt.xlabel("Number of Clusters(k)")
plt.ylabel("SSE")
plt.title("Elbow Method for Optimal K")
plt.show()

#find the most optimal num of clusters using the elbow method

#KMeans(init='random', n_clusters='8', n_init='10', random_state='42')  




