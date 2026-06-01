# Experiment 7: Rule-Based Classification using Weather Dataset

## Aim

To implement a Rule-Based Classification System using the Weather Dataset and generate simple IF–THEN rules for weather prediction using a Decision Tree Classifier.

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

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn import tree
```

---

## Theory

### What is Rule-Based Classification?

Rule-Based Classification is a supervised learning technique that classifies data using a set of IF–THEN rules.

Example:

```text
IF Temperature > 30°C
AND Humidity > 80%

THEN Weather = Rainy
```

The generated rules are easy to understand and explain, making them useful in expert systems and decision-support applications.

---

## What is a Decision Tree?

A Decision Tree is a machine learning algorithm that creates a tree-like structure of decisions.

Each path from:

* Root Node
* Internal Nodes
* Leaf Node

forms an IF–THEN classification rule.

Example:

```text
IF Precipitation > 2 mm
THEN Rainy

ELSE Sunny
```

---

## Objectives

1. Load and explore the Weather Dataset.
2. Visualize weather-related features.
3. Create a rule-based target variable.
4. Train a Decision Tree classifier.
5. Generate weather prediction rules.
6. Evaluate model performance.
7. Visualize decision rules.
8. Extract human-readable IF–THEN rules.

---

## Dataset Description

The dataset contains weather measurements collected over time.

### Features

| Feature          | Description             |
| ---------------- | ----------------------- |
| Temperature_C    | Temperature in Celsius  |
| Humidity_pct     | Relative Humidity (%)   |
| Precipitation_mm | Rainfall in millimeters |
| Wind_Speed_kmh   | Wind speed in km/h      |

---

## Methodology

### Step 1: Load Dataset

Read the Weather dataset.

```python
path = r"C:\Users\rjesh\weather_data.csv"

df = pd.read_csv(path)
```

Display first few records:

```python
print(df.head())
```

---

### Step 2: Explore Dataset

Check dataset dimensions.

```python
print(df.shape)
```

Display column names.

```python
print(df.columns)
```

View dataset information.

```python
print(df.info())
```

---

## Step 3: Data Visualization

### Temperature Distribution

```python
sns.histplot(
    df['Temperature_C'],
    bins=30,
    kde=True
)
```

### Observation

Shows the frequency distribution of temperature values.

---

### Humidity Distribution

```python
sns.histplot(
    df['Humidity_pct'],
    bins=30,
    kde=True
)
```

### Observation

Displays humidity variation across observations.

---

### Precipitation Distribution

```python
sns.histplot(
    df['Precipitation_mm'],
    bins=30,
    kde=True
)
```

### Observation

Shows rainfall occurrence and intensity.

---

### Wind Speed Distribution

```python
sns.histplot(
    df['Wind_Speed_kmh'],
    bins=30,
    kde=True
)
```

### Observation

Visualizes wind speed patterns within the dataset.

---

## Step 4: Create Target Variable

The dataset does not contain predefined weather labels.

Therefore, a rule-based target variable is created.

### Rule

```text
IF Precipitation > 2 mm

THEN Rainy

ELSE Sunny
```

Implementation:

```python
df['Weather_Type'] = np.where(
    df['Precipitation_mm'] > 2,
    'Rainy',
    'Sunny'
)
```

Display sample output:

```python
print(
    df[
        ['Precipitation_mm',
         'Weather_Type']
    ].head()
)
```

---

## Visualize Target Classes

```python
sns.countplot(
    x=df['Weather_Type']
)
```

### Observation

Displays the number of:

* Rainy Days
* Sunny Days

in the dataset.

---

## Step 5: Select Features and Target

### Input Features

```python
X = df[
    [
        'Temperature_C',
        'Humidity_pct',
        'Precipitation_mm',
        'Wind_Speed_kmh'
    ]
]
```

### Target Variable

```python
y = df['Weather_Type']
```

---

## Step 6: Label Encoding

Machine learning algorithms require numerical labels.

Convert:

| Label | Encoded Value |
| ----- | ------------- |
| Rainy | 0             |
| Sunny | 1             |

```python
encoder = LabelEncoder()

y = encoder.fit_transform(y)
```

Display encoded values:

```python
print(y[:10])
```

---

## Step 7: Train Rule-Based Classifier

### Algorithm Used

Decision Tree Classifier

```python
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    random_state=42
)
```

Train model:

```python
model.fit(X, y)
```

---

## Parameter Explanation

| Parameter           | Meaning                         |
| ------------------- | ------------------------------- |
| criterion='entropy' | Uses entropy for node splitting |
| max_depth=4         | Limits tree complexity          |
| random_state=42     | Produces reproducible results   |

---

## Step 8: Predictions

Generate predictions.

```python
y_pred = model.predict(X)
```

---

## Step 9: Accuracy Calculation

### Formula

```text
Accuracy =
Correct Predictions
-------------------
Total Predictions
```

Compute accuracy:

```python
accuracy = accuracy_score(
    y,
    y_pred
)
```

Display:

```python
print("Accuracy:", accuracy)
```

---

## Step 10: Confusion Matrix

A Confusion Matrix compares actual and predicted labels.

```python
cm = confusion_matrix(
    y,
    y_pred
)
```

Visualize:

```python
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)
```

---

## Confusion Matrix Interpretation

| Actual | Predicted | Meaning |
| ------ | --------- | ------- |
| Rainy  | Rainy     | Correct |
| Rainy  | Sunny     | Error   |
| Sunny  | Rainy     | Error   |
| Sunny  | Sunny     | Correct |

---

## Step 11: Classification Report

Generate detailed evaluation metrics.

```python
print(
    classification_report(
        y,
        y_pred
    )
)
```

Metrics include:

* Precision
* Recall
* F1 Score
* Support

---

## Step 12: Visualize Decision Tree Rules

This is the most important step in rule-based learning.

```python
tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Rainy','Sunny'],
    filled=True
)
```

### Observation

The tree structure displays all decision paths used for classification.

Each path forms a rule.

---

## Step 13: Extract IF–THEN Rules

Generate human-readable rules.

```python
rules = tree.export_text(
    model,
    feature_names=list(X.columns)
)

print(rules)
```

Example Rule:

```text
IF Precipitation_mm <= 2
THEN Sunny

IF Precipitation_mm > 2
THEN Rainy
```

---

## Results

* Weather dataset was loaded successfully.
* Weather classes were generated using precipitation-based rules.
* Decision Tree classifier was trained successfully.
* Prediction accuracy was calculated.
* Confusion Matrix and Classification Report were generated.
* Decision Tree visualization displayed classification logic.
* IF–THEN rules were extracted successfully.

---

## Advantages of Rule-Based Classification

* Easy to understand
* Human-readable rules
* Transparent decision-making
* Useful for expert systems
* Easy interpretation of predictions

---

## Limitations

* Can overfit large datasets
* Sensitive to noisy data
* Complex trees may become difficult to interpret
* Performance depends on rule quality

---

## Applications

* Weather Forecasting
* Medical Diagnosis
* Fraud Detection
* Customer Segmentation
* Expert Systems
* Business Decision Support

---

## Conclusion

The Decision Tree classifier successfully generated rule-based classification rules from weather data. By creating a target variable based on precipitation levels, the model learned interpretable IF–THEN rules and achieved high classification accuracy. The experiment demonstrated how rule-based systems can provide transparent and explainable predictions while maintaining good predictive performance.

---

## Outcome

A Rule-Based Classification System was successfully developed using a Decision Tree Classifier. The model generated human-readable IF–THEN rules for weather prediction and demonstrated the effectiveness of decision trees in explainable artificial intelligence applications.

---

## Viva Questions

### Q1. What is Rule-Based Classification?

Rule-Based Classification predicts classes using IF–THEN logical rules.

### Q2. Why is a Decision Tree considered a rule-based algorithm?

Because every path from the root node to a leaf node forms a classification rule.

### Q3. What is entropy?

Entropy measures impurity or randomness in a dataset.

### Q4. Why was Label Encoding used?

Machine learning algorithms require numerical values instead of text labels.

### Q5. What is the purpose of a Confusion Matrix?

It compares actual and predicted classifications.

### Q6. What is overfitting?

Overfitting occurs when a model learns training data too closely and performs poorly on new data.

### Q7. What is a leaf node?

A leaf node represents the final predicted class in a decision tree.

### Q8. What is the advantage of rule-based systems?

They provide transparent, interpretable, and explainable predictions.

