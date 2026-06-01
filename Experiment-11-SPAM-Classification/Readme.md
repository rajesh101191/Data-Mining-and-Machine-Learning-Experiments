# Experiment 11: Spam Message Classification Using Naive Bayes

## Aim

To build a Spam Message Classification System using the Naive Bayes algorithm and classify SMS messages as Spam or Ham (Non-Spam).

---

## Software Required

* Python 3.x
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* WordCloud

---

## Libraries Used

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
```

---

# Theory

## What is Spam Classification?

Spam Classification is a Machine Learning application used to automatically classify text messages into:

* Ham (Legitimate Message)
* Spam (Unwanted or Fraudulent Message)

Spam filtering is widely used in:

* SMS Applications
* Email Services
* Messaging Platforms
* Cybersecurity Systems

---

## What is Naive Bayes?

Naive Bayes is a supervised machine learning algorithm based on Bayes' Theorem.

It is particularly effective for:

* Text Classification
* Spam Detection
* Sentiment Analysis
* Document Categorization

The algorithm assumes that features are independent of one another.

---

## Objectives

1. Load and preprocess SMS Spam Collection Dataset.
2. Convert text messages into numerical features.
3. Train a Naive Bayes classifier.
4. Classify messages as Spam or Ham.
5. Evaluate model performance.
6. Visualize classification results.
7. Test custom messages for spam prediction.

---

## Dataset Description

The SMS Spam Collection Dataset contains:

| Column  | Description      |
| ------- | ---------------- |
| label   | ham or spam      |
| message | SMS text message |

### Class Labels

| Label | Meaning                      |
| ----- | ---------------------------- |
| ham   | Legitimate Message           |
| spam  | Unwanted/Promotional Message |

---

# Methodology

## Step 1: Load Dataset

Read SMS Spam Collection Dataset.

```python
path = r"C:\Users\rjesh\SMSSpamCollection.csv"

with open(path, 'r', encoding='latin-1') as f:
    lines = f.readlines()
```

Display sample information.

```python
print("Total lines:", len(lines))
print("Sample line:", lines[0])
```

### Observation

The dataset contains SMS messages labeled as spam or ham.

---

## Step 2: Create DataFrame

Separate labels and messages.

```python
data = []

for line in lines:
    parts = line.strip().split('\t',1)

    if len(parts)==2:
        label = parts[0]
        message = parts[1]

        data.append([label,message])

df = pd.DataFrame(
    data,
    columns=['label','message']
)
```

Display first records.

```python
print(df.head())
```

---

## Step 3: Explore Dataset

Display class distribution.

```python
print(df['label'].value_counts())
```

Display dataset shape.

```python
print(df.shape)
```

### Observation

The dataset contains both spam and ham messages.

---

## Step 4: Data Preprocessing

Convert text labels into numerical format.

```python
df['label'] = df['label'].map(
    {
        'ham':0,
        'spam':1
    }
)
```

### Encoding

| Original Label | Encoded Value |
| -------------- | ------------- |
| Ham            | 0             |
| Spam           | 1             |

---

## Step 5: Define Features and Target

```python
X = df['message']

y = df['label']
```

---

## Step 6: Train-Test Split

Split dataset into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Parameter Description

| Parameter       | Meaning              |
| --------------- | -------------------- |
| test_size=0.2   | 20% Testing Data     |
| random_state=42 | Reproducible Results |

---

## Step 7: Feature Extraction

Machine learning algorithms cannot process raw text directly.

Use CountVectorizer to convert text into numerical vectors.

```python
vectorizer = CountVectorizer(
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(
    X_train
)

X_test_vec = vectorizer.transform(
    X_test
)
```

### Functions of CountVectorizer

* Tokenization
* Vocabulary Creation
* Word Frequency Counting
* Text Vectorization

---

## Step 8: Train Naive Bayes Model

Create classifier.

```python
model = MultinomialNB()
```

Train classifier.

```python
model.fit(
    X_train_vec,
    y_train
)
```

### Why Multinomial Naive Bayes?

Because:

* Works well for text classification.
* Handles word frequency data efficiently.
* Fast and memory efficient.

---

## Step 9: Make Predictions

Predict message categories.

```python
y_pred = model.predict(
    X_test_vec
)
```

---

## Step 10: Accuracy Evaluation

Calculate model accuracy.

```python
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)
```

### Accuracy Formula

```text
Accuracy =
Correct Predictions
-------------------
Total Predictions
```

### Interpretation

Higher accuracy indicates better spam detection performance.

---

## Step 11: Confusion Matrix

Generate confusion matrix.

```python
cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)
```

### Interpretation

| Actual | Predicted | Meaning        |
| ------ | --------- | -------------- |
| Ham    | Ham       | True Negative  |
| Ham    | Spam      | False Positive |
| Spam   | Ham       | False Negative |
| Spam   | Spam      | True Positive  |

---

## Step 12: Classification Report

Generate detailed classification metrics.

```python
print(
    classification_report(
        y_test,
        y_pred
    )
)
```

Metrics include:

* Precision
* Recall
* F1-Score
* Support

---

## Performance Metrics

### Precision

Measures how many predicted spam messages are actually spam.

Formula:

```text
Precision =
TP
--------
TP + FP
```

### Recall

Measures how many actual spam messages are correctly detected.

Formula:

```text
Recall =
TP
--------
TP + FN
```

### F1-Score

Harmonic mean of Precision and Recall.

Formula:

```text
F1 =
2 × Precision × Recall
----------------------
Precision + Recall
```

---

## Step 13: Sample Prediction

Test custom messages.

```python
messages = [
    "Congratulations! You won a free iPhone.",
    "Hey, are we meeting tomorrow?",
    "Urgent! Verify your bank account immediately."
]

msg_vec = vectorizer.transform(
    messages
)

pred = model.predict(msg_vec)
```

Display predictions.

```python
for m,p in zip(messages,pred):

    print(m)

    print(
        "SPAM"
        if p==1
        else "HAM"
    )
```

### Observation

The trained model can classify unseen SMS messages.

---

# Visualization

## 1. Class Distribution

```python
sns.countplot(
    x=df['label']
)
```

### Observation

Shows the number of spam and ham messages.

---

## 2. Confusion Matrix Heatmap

```python
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)
```

### Observation

Provides visual representation of classification accuracy.

---

## 3. Spam Word Cloud

```python
spam_words = " ".join(
    df[df['label']==1]['message']
)

WordCloud().generate(spam_words)
```

### Observation

Displays frequently occurring words in spam messages.

---

## 4. Ham Word Cloud

```python
ham_words = " ".join(
    df[df['label']==0]['message']
)

WordCloud().generate(ham_words)
```

### Observation

Displays commonly used words in genuine messages.

---

## 5. Top Spam Words

```python
from collections import Counter
```

Count and visualize most common spam words.

### Observation

Identifies words frequently used by spammers.

---

## 6. Model Accuracy Visualization

```python
plt.bar(
    ['Naive Bayes'],
    [accuracy]
)
```

### Observation

Displays overall classification accuracy.

---

# Results

* SMS Spam Collection Dataset was successfully loaded.
* Text messages were converted into numerical vectors.
* Multinomial Naive Bayes classifier was trained successfully.
* Spam and ham messages were classified accurately.
* Performance was evaluated using accuracy, confusion matrix, precision, recall, and F1-score.
* Visualizations provided deeper insight into message characteristics.

---

# Applications

* SMS Spam Filtering
* Email Spam Detection
* Fraud Message Detection
* Customer Support Automation
* Social Media Content Filtering
* Cybersecurity Systems

---

# Advantages

* Fast Training
* Efficient Text Classification
* High Accuracy
* Simple Implementation
* Works Well with Large Text Datasets

---

# Limitations

* Assumes Feature Independence
* Sensitive to Rare Words
* Limited Context Understanding
* Performance Depends on Training Data Quality

---

# Conclusion

The Naive Bayes Spam Message Classification model was successfully implemented using the SMS Spam Collection Dataset. The classifier effectively distinguished spam messages from legitimate messages and achieved high classification accuracy. The experiment demonstrated how Natural Language Processing and Machine Learning can be combined to build practical spam filtering systems used in real-world communication platforms.

---

# Outcome

A complete Spam Message Classification System was developed using Multinomial Naive Bayes. The model successfully classified SMS messages as Spam or Ham and demonstrated the effectiveness of probabilistic machine learning techniques for text classification.

---

# Viva Questions

### Q1. What is spam classification?

Spam classification is the process of identifying unwanted messages automatically.

### Q2. Why is Naive Bayes widely used for spam detection?

Because it is fast, efficient, and performs well on text data.

### Q3. What is CountVectorizer?

CountVectorizer converts text into numerical word-frequency vectors.

### Q4. What is the role of feature extraction?

It transforms text into a machine-readable numerical format.

### Q5. What is Multinomial Naive Bayes?

A Naive Bayes variant designed for discrete count data such as word frequencies.

### Q6. What is precision?

Precision measures the correctness of positive predictions.

### Q7. What is recall?

Recall measures the ability to detect all actual positive instances.

### Q8. What is F1-score?

F1-score is the harmonic mean of precision and recall.

### Q9. What is a confusion matrix?

A table showing actual versus predicted classifications.

### Q10. Give one real-world application of spam classification.

Email spam filtering used by mail services such as Gmail and Outlook.

