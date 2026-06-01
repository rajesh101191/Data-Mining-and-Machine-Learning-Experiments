# Experiment 7: DBSCAN Clustering Using Moons Dataset

## Aim

To implement the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clustering algorithm using the synthetic Moons Dataset and identify clusters and noise points.

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
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
```

---

## Theory

### What is DBSCAN?

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised machine learning algorithm that groups closely packed data points together while identifying isolated points as noise or outliers.

Unlike K-Means Clustering:

* DBSCAN does not require the number of clusters beforehand.
* Can detect clusters of arbitrary shapes.
* Effectively identifies noise and outliers.
* Works well for non-linear datasets.

---

## Key Concepts in DBSCAN

| Concept       | Description                                                          |
| ------------- | -------------------------------------------------------------------- |
| Core Point    | A point having sufficient neighboring points within a radius         |
| Border Point  | A point near a core point but with fewer neighbors                   |
| Noise Point   | An isolated point that does not belong to any cluster                |
| Epsilon (eps) | Radius defining the neighborhood around a point                      |
| min_samples   | Minimum number of neighboring points required to form a dense region |

---

## Objectives

1. Generate a synthetic Moons Dataset.
2. Visualize the original dataset.
3. Apply DBSCAN clustering.
4. Identify clusters and noise points.
5. Visualize clustered data.
6. Evaluate clustering quality using Silhouette Score.
7. Compare original and clustered datasets.

---

## Methodology

### Step 1: Generate Moons Dataset

The synthetic dataset is generated using Scikit-learn.

```python
X, y = make_moons(
    n_samples=500,
    noise=0.08,
    random_state=42
)
```

### Parameter Description

| Parameter       | Meaning                        |
| --------------- | ------------------------------ |
| n_samples=500   | Total number of data points    |
| noise=0.08      | Adds randomness to the dataset |
| random_state=42 | Ensures reproducible results   |

---

### Step 2: Visualize Original Dataset

Plot the generated moons dataset.

```python
plt.scatter(
    X[:,0],
    X[:,1]
)
```

### Observation

The dataset contains:

* Two moon-shaped clusters.
* Non-linear structures.
* Some noisy points.

This dataset is ideal for DBSCAN because traditional clustering methods such as K-Means struggle with non-linear cluster boundaries.

---

### Step 3: Apply DBSCAN Algorithm

Create and fit the DBSCAN model.

```python
dbscan = DBSCAN(
    eps=0.2,
    min_samples=5
)

clusters = dbscan.fit_predict(X)
```

---

## Parameter Explanation

| Parameter     | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| eps=0.2       | Radius used to search neighboring points                     |
| min_samples=5 | Minimum neighboring points required to create a dense region |

---

### Step 4: Check Cluster Labels

Display unique cluster labels.

```python
print(np.unique(clusters))
```

### Label Interpretation

| Label | Meaning      |
| ----- | ------------ |
| 0     | Cluster 1    |
| 1     | Cluster 2    |
| -1    | Noise Points |

---

### Step 5: Visualize DBSCAN Clusters

Visualize clustered data.

```python
plt.scatter(
    X[:,0],
    X[:,1],
    c=clusters,
    cmap='viridis'
)
```

### Observation

The plot displays:

* Different colors representing clusters.
* Clearly separated moon-shaped groups.
* Noise points assigned separate labels.

---

### Step 6: Highlight Noise Points

DBSCAN marks noise points using label:

```text
-1
```

Extract noise points:

```python
noise_points = X[clusters == -1]
```

Plot noise separately.

```python
plt.scatter(
    noise_points[:,0],
    noise_points[:,1],
    color='red'
)
```

### Observation

Noise points represent outliers that do not belong to any cluster.

---

### Step 7: Determine Number of Clusters

Calculate cluster count excluding noise.

```python
n_clusters = len(
    set(clusters)
) - (
    1 if -1 in clusters else 0
)
```

Display result:

```python
print("Number of Clusters:", n_clusters)
```

---

### Step 8: Count Noise Points

Determine total outliers.

```python
noise_count = list(clusters).count(-1)

print("Number of Noise Points:", noise_count)
```

---

### Step 9: Calculate Silhouette Score

Evaluate clustering quality.

```python
score = silhouette_score(
    X,
    clusters
)

print("Silhouette Score:", score)
```

### Interpretation

| Score Value | Meaning              |
| ----------- | -------------------- |
| Near +1     | Excellent clustering |
| Near 0      | Overlapping clusters |
| Near -1     | Poor clustering      |

Higher values indicate better cluster separation.

---

### Step 10: Compare Original vs Clustered Data

Visual comparison between:

1. Original Dataset
2. DBSCAN Clustered Dataset

```python
plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1])

plt.subplot(1,2,2)
plt.scatter(
    X[:,0],
    X[:,1],
    c=clusters
)
```

### Observation

DBSCAN successfully identifies cluster structures and separates noise points from meaningful groups.

---

## Results

* A synthetic Moons Dataset containing 500 samples was generated successfully.
* DBSCAN clustering was applied using eps = 0.2 and min_samples = 5.
* Two moon-shaped clusters were accurately identified.
* Noise points were successfully detected and labeled.
* Cluster visualization confirmed proper separation of non-linear structures.
* Silhouette Score indicated good clustering quality.
* DBSCAN effectively handled complex cluster shapes without requiring the number of clusters beforehand.

---

## Advantages of DBSCAN

* Detects arbitrary-shaped clusters.
* No need to specify the number of clusters.
* Handles noise and outliers efficiently.
* Effective for non-linear datasets.
* Robust to cluster shape variations.

---

## Limitations of DBSCAN

* Sensitive to eps parameter selection.
* Performance may degrade for varying density datasets.
* Less effective in very high-dimensional spaces.
* Parameter tuning can be challenging.

---

## Conclusion

The DBSCAN clustering algorithm was successfully implemented on the Moons Dataset. The algorithm accurately identified the two non-linear moon-shaped clusters and detected noise points automatically. Unlike K-Means, DBSCAN effectively handled arbitrary cluster shapes without requiring prior knowledge of the number of clusters. The experiment demonstrates the usefulness of density-based clustering techniques for real-world datasets containing complex structures and outliers.

---

## Outcome

A density-based clustering model was successfully developed using DBSCAN. The model effectively segmented non-linear data into meaningful clusters while identifying noise points, demonstrating the advantages of DBSCAN over traditional centroid-based clustering methods.

---

## Viva Questions

### Q1. What is DBSCAN?

DBSCAN is a density-based clustering algorithm used to group closely packed data points and identify outliers.

### Q2. What is the role of eps?

The eps parameter defines the neighborhood radius around a data point.

### Q3. What are noise points?

Noise points are isolated points that do not belong to any cluster and are labeled as -1.

### Q4. Why is DBSCAN better than K-Means for the Moons Dataset?

Because DBSCAN can detect non-linear and arbitrarily shaped clusters, whereas K-Means assumes spherical clusters.

### Q5. What does label -1 represent?

Label -1 represents noise or outlier points detected by DBSCAN.

### Q6. What is a Core Point?

A core point is a point that has at least min_samples neighboring points within the eps radius.

### Q7. What is a Border Point?

A border point lies within the neighborhood of a core point but does not itself have enough neighbors to be a core point.

### Q8. What is the significance of Silhouette Score?

Silhouette Score measures how well data points fit within their assigned clusters and how distinct the clusters are from each other.

