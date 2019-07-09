# cluster

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
x = [1,22,3,4,58,78,90]
y = [2,20,9,16,106,80,90]

x = np.array([[1,2],[3,9],[4,16],[59,106],[78,80],[80,90]])

model = KMeans(n_clusters=2)
model.fit(x)
centroids = model.cluster_centers_
print("centroids",centroids)
labels=model.labels_
print(labels)
for i in range(len(x)):
    print("points",x[i],"is having cluster number",labels[i])

