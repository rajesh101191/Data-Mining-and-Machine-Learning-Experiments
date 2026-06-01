# Experiment 6: K-Means Clustering Using Mall Customers Dataset

## Aim

To implement the K-Means Clustering Algorithm for customer segmentation using the Mall Customers dataset and analyze customer groups based on annual income and spending behavior.

---

## Software Required

* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Libraries Used

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
```

---

## Theory

### What is K-Means Clustering?

K-Means is an unsupervised machine learning algorithm used to partition data into K distinct clusters based on similarity.

The algorithm works by:

1. Selecting K initial centroids.
2. Calculating distances between data points and centroids.
3. Assigning points to the nearest centroid.
4. Updating centroid positions.
5. Repeating the process until convergence.

---

## Applications of K-Means

* Customer Segmentation
* Market Basket Analysis
* Recommendation Systems
* Image Compression
* Pattern Recognition
* Business Intelligence

---

## Dataset Description

The Mall Customers dataset contains information about customers visiting a shopping mall.

### Features

| Feature                | Description                       |
| ---------------------- | --------------------------------- |
| CustomerID             | Unique customer identifier        |
| Gender                 | Male/Female                       |
| Age                    | Customer age                      |
| Annual Income (k$)     | Annual income in thousand dollars |
| Spending Score (1-100) | Customer spending behavior score  |

---

## Objectives

1. Load and explore the Mall Customers dataset.
2. Visualize customer characteristics.
3. Select suitable features for clustering.
4. Scale numerical features.
5. Determine the optimal number of clusters using the Elbow Method.
6. Apply K-Means clustering.
7. Visualize customer segments.
8. Evaluate clustering quality using Silhouette Score.
9. Analyze cluster-wise customer behavior.

---

## Methodology

### Step 1: Load Dataset

Load the dataset using Pandas.

```python
df = pd.read_csv("Mall_Customers.csv")
```

Display first few records:

```python
df.head()
```

---

### Step 2: Explore Dataset

Understand dataset structure using:

```python
df.shape
df.columns
df.info()
```

This provides:

* Number of records
* Number of features
* Data types
* Missing values

---

### Step 3: Data Visualization

#### Annual Income Distribution

```python
sns.histplot(
    df['Annual Income (k$)'],
    bins=20,
    kde=True
)
```

Purpose:

* Understand income distribution.
* Identify skewness and spread.

---

#### Spending Score Distribution

```python
sns.histplot(
    df['Spending Score (1-100)'],
    bins=20,
    kde=True
)
```

Purpose:

* Analyze customer spending patterns.
* Identify concentration of spending scores.

---

#### Income vs Spending Scatter Plot

```python
plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)']
)
```

Purpose:

* Visualize customer distribution.
* Identify potential clusters.

---

### Step 4: Feature Selection

The following features are selected:

* Annual Income (k$)
* Spending Score (1-100)

```python
X = df[
    [
        'Annual Income (k$)',
        'Spending Score (1-100)'
    ]
]
```

These attributes are highly useful for customer segmentation.

---

### Step 5: Feature Scaling

Since clustering is distance-based, feature scaling is necessary.

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

### Why Scaling?

* Income and spending score have different ranges.
* Prevents one feature from dominating distance calculations.
* Improves clustering performance.

---

### Step 6: Determine Optimal Clusters Using Elbow Method

#### WCSS (Within Cluster Sum of Squares)

WCSS measures cluster compactness.

Lower WCSS indicates tighter clusters.

```python
wcss = []

for i in range(1,11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)
```

---

### Elbow Plot

```python
plt.plot(
    range(1,11),
    wcss,
    marker='o'
)
```

Observation:

The "elbow point" represents the optimal number of clusters.

For the Mall Customers dataset:

```text
K = 5
```

is typically selected.

---

### Step 7: Apply K-Means Clustering

Create the K-Means model.

```python
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
```

Generate cluster labels:

```python
clusters = kmeans.fit_predict(X_scaled)
```

---

### Step 8: Add Cluster Labels

Store cluster assignments.

```python
df['Cluster'] = clusters
```

Each customer is assigned to one of five clusters:

* Cluster 0
* Cluster 1
* Cluster 2
* Cluster 3
* Cluster 4

---

### Step 9: Visualize Clusters

```python
plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)
```

Purpose:

* Visualize customer segments.
* Observe separation between clusters.

---

### Step 10: Visualize Centroids

Convert centroids back to original scale.

```python
centroids = scaler.inverse_transform(
    kmeans.cluster_centers_
)
```

Plot cluster centers.

```python
plt.scatter(
    centroids[:,0],
    centroids[:,1],
    marker='X'
)
```

### What are Centroids?

Centroids represent:

* Center point of a cluster.
* Average position of all points within a cluster.

K-Means continuously updates centroids until convergence.

---

### Step 11: Silhouette Score

Silhouette Score evaluates clustering quality.

```python
score = silhouette_score(
    X_scaled,
    clusters
)
```

Interpretation:

| Score   | Meaning              |
| ------- | -------------------- |
| Near +1 | Excellent clustering |
| Near 0  | Overlapping clusters |
| Near -1 | Poor clustering      |

Higher values indicate better cluster separation.

---

### Step 12: Cluster-wise Customer Count

```python
df['Cluster'].value_counts()
```

Purpose:

* Determine number of customers in each segment.
* Analyze cluster distribution.

---

### Step 13: Cluster-wise Box Plot

```python
sns.boxplot(
    x='Cluster',
    y='Spending Score (1-100)',
    data=df
)
```

Purpose:

* Compare spending behavior across clusters.
* Identify high-value customer groups.

---

## Results

* Mall customer data was successfully loaded and analyzed.
* Income and spending behavior were selected as clustering features.
* Features were standardized using StandardScaler.
* The Elbow Method identified the optimal number of clusters.
* K-Means clustering successfully segmented customers into five groups.
* Cluster visualization clearly displayed customer segments.
* Centroid visualization highlighted cluster centers.
* Silhouette Score confirmed clustering quality.
* Customer behavior patterns were effectively identified.

---

## Advantages of K-Means

* Simple and easy to implement.
* Fast execution.
* Scales well to large datasets.
* Easy visualization and interpretation.
* Widely used in business analytics.

---

## Limitations of K-Means

* Requires predefined value of K.
* Sensitive to outliers.
* May converge to local minima.
* Works best with spherical clusters.

---

## Conclusion

The K-Means Clustering algorithm was successfully implemented on the Mall Customers dataset. Customers were segmented into meaningful groups based on annual income and spending score. The Elbow Method and Silhouette Score helped evaluate clustering quality. The resulting customer segments can support targeted marketing and business decision-making.

---

## Outcome

A customer segmentation model was developed using K-Means Clustering. The experiment demonstrated how unsupervised learning can identify hidden patterns within customer data and group customers with similar purchasing behavior.

---

## Viva Questions

### Q1. What is K-Means Clustering?

K-Means is an unsupervised learning algorithm that groups similar data points into clusters.

### Q2. What is a centroid?

A centroid is the center point representing a cluster.

### Q3. What is WCSS?

WCSS (Within Cluster Sum of Squares) measures the compactness of clusters.

### Q4. Why is the Elbow Method used?

The Elbow Method helps determine the optimal number of clusters.

### Q5. Why is feature scaling important in K-Means?

Scaling ensures all features contribute equally to distance calculations.

