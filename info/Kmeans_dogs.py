#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Load the dataset
df = pd.read_csv('dogs_cleaned.csv')
print(df)

# Use data to determine clusters
print(df.describe())

#Select only numerical and scale
numeric_df = df.select_dtypes(include=['number', 'bool'])
scaler = StandardScaler()
scaled_df = StandardScaler().fit_transform(numeric_df)

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
plt.plot(range(1, 5), sse)
plt.xticks(range(1, 5))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

#find the most optimal num of clusters using the elbow method

#KMeans(init='random', n_clusters='8', n_init='10', random_state='42')   




