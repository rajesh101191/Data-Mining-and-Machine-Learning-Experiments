# Experiment 6: K-Nearest Neighbor (KNN) Using Social Network Ads Dataset

## Aim

To implement the K-Nearest Neighbor (KNN) algorithm using the Social Network Ads dataset and classify whether a customer will purchase a product based on Age and Estimated Salary.

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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)
```

---

## Theory

### What is K-Nearest Neighbor (KNN)?

K-Nearest Neighbor (KNN) is a supervised machine learning algorithm used for classification and regression problems.

KNN works on the assumption that:

> Similar data points tend to exist close to one another.

For classification tasks, the algorithm predicts the class of a new data point based on the majority class among its K nearest neighbors.

### Advantages of KNN

* Simple and easy to implement.
* No training phase required.
* Effective for small and medium-sized datasets.
* Works well for nonlinear decision boundaries.

### Limitations of KNN

* Computationally expensive for large datasets.
* Sensitive to feature scaling.
* Performance depends on the choice of K.

---

## Dataset Description

The Social Network Ads dataset contains customer information collected from social media advertisements.

### Attributes

| Feature         | Description                       |
| --------------- | --------------------------------- |
| Age             | Customer age                      |
| EstimatedSalary | Annual salary of customer         |
| Purchased       | Target variable (0 = No, 1 = Yes) |

---

## Objectives

1. Load and explore the Social Network Ads dataset.
2. Perform preprocessing and feature selection.
3. Split data into training and testing sets.
4. Scale numerical features.
5. Train a KNN classifier.
6. Predict customer purchasing behavior.
7. Evaluate model performance using classification metrics.
8. Visualize classification results and decision boundaries.
9. Save prediction results.

---

## Methodology

### Step 1: Data Loading

The dataset is loaded using Pandas.

```python
df = pd.read_csv("Social_Network_Ads.csv")
```

Display first few records:

```python
df.head()
```

---

### Step 2: Dataset Exploration

Basic dataset information is obtained using:

```python
df.shape
df.columns
df.info()
```

This helps understand:

* Number of records
* Number of features
* Data types
* Missing values

---

### Step 3: Data Preprocessing

Check for missing values:

```python
df.isnull().sum()
```

Since all features are already numerical, no encoding is required.

---

### Step 4: Feature Selection

Select important features for classification.

#### Input Features

* Age
* EstimatedSalary

#### Target Variable

* Purchased

```python
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
```

---

### Step 5: Train-Test Split

The dataset is divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)
```

Dataset Distribution:

* 75% Training Data
* 25% Testing Data

---

### Step 6: Feature Scaling

KNN is based on distance calculations.

Since salary values are much larger than age values, scaling is essential.

StandardScaler is used:

```python
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

Benefits:

* Prevents large-valued features from dominating.
* Improves prediction accuracy.
* Enhances distance calculations.

---

### Step 7: Create KNN Classifier

Create a KNN model with:

```python
model = KNeighborsClassifier(
    n_neighbors=5
)
```

Where:

* K = 5
* Euclidean distance is used by default.

---

### Step 8: Train the Model

The classifier learns patterns from the training data.

```python
model.fit(
    X_train,
    y_train
)
```

---

### Step 9: Prediction

Predict customer purchasing behavior.

```python
y_pred = model.predict(X_test)
```

Output:

```python
print(y_pred)
```

---

## Model Evaluation

### 1. Confusion Matrix

A confusion matrix compares actual and predicted values.

```python
cm = confusion_matrix(
    y_test,
    y_pred
)
```

| Actual / Predicted | Positive | Negative |
| ------------------ | -------- | -------- |
| Positive           | TP       | FN       |
| Negative           | FP       | TN       |

Where:

* TP = True Positive
* TN = True Negative
* FP = False Positive
* FN = False Negative

---

### 2. Accuracy Score

Measures overall correctness of predictions.

```python
accuracy = accuracy_score(
    y_test,
    y_pred
)
```

Formula:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Higher accuracy indicates better classification performance.

---

### 3. Classification Report

Provides:

* Precision
* Recall
* F1-Score
* Support

```python
classification_report(
    y_test,
    y_pred
)
```

Interpretation:

* Precision measures prediction correctness.
* Recall measures ability to detect positive cases.
* F1 Score balances precision and recall.

---

## Actual vs Predicted Comparison

Create comparison table:

```python
comparison = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
```

This helps compare model predictions with actual customer decisions.

---

## Visualizations

### 1. Customer Purchase Distribution

Scatter plot of:

* Age
* Estimated Salary

Colored by purchase status.

```python
plt.scatter(
    df['Age'],
    df['EstimatedSalary'],
    c=df['Purchased']
)
```

Observation:

Different customer groups can be visually identified.

---

### 2. Actual vs Predicted Purchase Decisions

```python
plt.plot(
    comparison['Actual']
)
plt.plot(
    comparison['Predicted']
)
```

Observation:

Predicted values should closely follow actual values.

---

### 3. Confusion Matrix Heatmap

```python
sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)
```

Benefits:

* Easy interpretation of classification performance.
* Highlights correctly and incorrectly classified samples.

---

### 4. Purchase Distribution Bar Chart

```python
df['Purchased'].value_counts().plot(
    kind='bar'
)
```

Shows:

* Number of customers who purchased.
* Number of customers who did not purchase.

---

### 5. Age vs Salary Colored by Purchase

```python
plt.scatter(
    df['Age'],
    df['EstimatedSalary'],
    c=df['Purchased']
)
```

Observation:

Customer purchasing patterns become visually apparent.

---

### 6. Decision Boundary Visualization (Training Set)

Decision boundaries show how KNN separates classes.

```python
plt.contourf(...)
```

Observation:

Regions represent predicted classes.

Training data points are plotted over these regions.

---

### 7. Decision Boundary Visualization (Test Set)

Decision boundaries are generated for testing data.

Observation:

Shows model generalization on unseen data.

---

## Saving Output

Prediction results are stored in a CSV file.

```python
comparison.to_csv(
    "knn_output.csv",
    index=False
)
```

---

## Results

* Social Network Ads dataset was successfully loaded and analyzed.
* Age and Estimated Salary were selected as input features.
* Data was standardized using StandardScaler.
* KNN classifier was trained with K = 5.
* Customer purchasing behavior was predicted successfully.
* Classification metrics demonstrated strong predictive performance.
* Decision boundary visualizations clearly illustrated class separation.
* Prediction results were saved successfully.

---

## Conclusion

The K-Nearest Neighbor (KNN) algorithm was implemented successfully using the Social Network Ads dataset. The model classified customer purchasing behavior based on Age and Estimated Salary with high accuracy. Evaluation metrics, confusion matrix analysis, and decision boundary visualizations demonstrated the effectiveness of KNN for classification problems.

---

## Outcome

A KNN classification model was developed and evaluated successfully. The model accurately predicts whether a customer will purchase a product based on demographic and salary information, demonstrating the practical application of supervised machine learning for customer behavior analysis.

