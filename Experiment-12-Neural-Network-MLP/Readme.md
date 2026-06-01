# Experiment 12: Neural Network Classification using MLP

## Aim

To implement a Multi-Layer Perceptron (MLP) Neural Network for handwritten digit classification using the MNIST dataset and evaluate its performance using accuracy, confusion matrix, classification report, and prediction visualization.

---

## Software Requirements

* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
```

---

## Theory

### What is a Neural Network?

A Neural Network is a machine learning model inspired by the human brain. It consists of interconnected neurons organized into layers.

### What is MLP?

Multi-Layer Perceptron (MLP) is a feedforward artificial neural network that contains:

* Input Layer
* One or More Hidden Layers
* Output Layer

MLP learns patterns through forward propagation and backpropagation.

---

## Objectives

1. Load the MNIST handwritten digit dataset.
2. Preprocess and normalize image data.
3. Train an MLP classifier.
4. Evaluate classification accuracy.
5. Visualize predictions and misclassified images.
6. Analyze network performance using confusion matrix and classification report.

---

## Dataset Description

### MNIST Dataset

The MNIST dataset contains handwritten digits from 0–9.

| Attribute    | Description                |
| ------------ | -------------------------- |
| Label        | Digit (0–9)                |
| Pixel Values | 784 pixel intensity values |
| Image Size   | 28 × 28 pixels             |

---

## Methodology

### Step 1: Load Dataset

```python
train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")
```

### Observation

The first column represents the digit label and the remaining 784 columns represent pixel values.

---

### Step 2: Explore Dataset

```python
print(train_df.shape)
print(test_df.shape)
```

### Observation

Each image consists of 784 pixels corresponding to a 28×28 grayscale image.

---

### Step 3: Separate Features and Labels

```python
X_train = train_df.iloc[:,1:]
y_train = train_df.iloc[:,0]

X_test = test_df.iloc[:,1:]
y_test = test_df.iloc[:,0]
```

---

### Step 4: Normalize Data

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

### Why Normalize?

* Faster convergence
* Improved training stability
* Better neural network performance

---

### Step 5: Visualize Sample Images

```python
image = X_train.iloc[0].values.reshape(28,28)

plt.imshow(image,cmap='gray')
```

### Observation

Visual inspection confirms images are loaded correctly.

---

### Step 6: Create MLP Model

```python
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    max_iter=20,
    random_state=42
)
```

### Parameter Description

| Parameter                 | Meaning                  |
| ------------------------- | ------------------------ |
| hidden_layer_sizes=(100,) | 100 hidden neurons       |
| activation='relu'         | ReLU activation function |
| solver='adam'             | Optimization algorithm   |
| max_iter=20               | Training iterations      |
| random_state=42           | Reproducibility          |

---

### Step 7: Train Neural Network

```python
mlp.fit(X_train,y_train)
```

### Observation

The model learns handwritten digit patterns through backpropagation.

---

### Training Loss Curve

```python
plt.plot(mlp.loss_curve_)
```

### Observation

Loss decreases as training progresses, indicating learning.

---

### Step 8: Predictions

```python
y_pred = mlp.predict(X_test)
```

### Observation

The trained model predicts digits from 0–9 on unseen images.

---

### Visualize Misclassified Images

```python
misclassified = np.where(y_test != y_pred)[0]
```

### Observation

Misclassified images help identify challenging digit patterns.

---

### Step 9: Accuracy Calculation

```python
accuracy = accuracy_score(
    y_test,
    y_pred
)

print(accuracy)
```

### Accuracy Formula

```text
Accuracy =
Correct Predictions
-------------------
Total Predictions
```

---

### Network Architecture

Input Layer:

* 784 Neurons

Hidden Layer:

* 100 Neurons

Output Layer:

* 10 Neurons (Digits 0–9)

---

### Visualizing Learned Features

```python
weights = mlp.coefs_[0]
```

### Observation

Hidden neurons learn digit edges, curves, and stroke patterns.

---

### Step 10: Confusion Matrix

```python
cm = confusion_matrix(
    y_test,
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

### Interpretation

* Diagonal values indicate correct classifications.
* Off-diagonal values indicate errors.

---

### Step 11: Classification Report

```python
print(
    classification_report(
        y_test,
        y_pred
    )
)
```

### Metrics

| Metric    | Meaning                         |
| --------- | ------------------------------- |
| Precision | Correct positive predictions    |
| Recall    | Ability to find positives       |
| F1-Score  | Balance of precision and recall |
| Support   | Number of samples               |

---

### Step 12: Visualize Predictions

```python
plt.imshow(
    X_test.iloc[0].values.reshape(28,28),
    cmap='gray'
)
```

### Observation

Confirms the model correctly recognizes handwritten digits.

---

## Results

* MNIST dataset was loaded successfully.
* Images were normalized and visualized.
* MLP neural network was trained successfully.
* High classification accuracy was achieved.
* Confusion matrix and classification report demonstrated strong performance.
* Learned neuron features and predictions were visualized successfully.

---

## Advantages of MLP

* Learns complex non-linear patterns.
* Effective for image classification.
* High prediction accuracy.
* Supports multi-class classification.

---

## Limitations

* Longer training time.
* Computationally intensive.
* Sensitive to hyperparameters.
* Requires sufficient training data.

---

## Applications

* Handwritten Digit Recognition
* Character Recognition
* Image Classification
* Face Recognition
* Medical Image Analysis
* Pattern Recognition

---

## Conclusion

The Multi-Layer Perceptron (MLP) Neural Network was successfully implemented for handwritten digit recognition using the MNIST dataset. The model effectively learned image patterns through hidden neurons and achieved high classification accuracy. The experiment demonstrated the capability of neural networks in solving image classification problems.

---

## Outcome

A neural network-based digit classification system was successfully developed using the MNIST dataset. The trained MLP model accurately classified handwritten digits and demonstrated the effectiveness of deep learning techniques for pattern recognition.

---

## Viva Questions

### Q1. What is MLP?

MLP is a feedforward artificial neural network containing input, hidden, and output layers.

### Q2. Why normalize data?

Normalization improves training speed and model convergence.

### Q3. What is ReLU?

ReLU is an activation function that returns positive values and helps avoid vanishing gradients.

### Q4. What is backpropagation?

Backpropagation updates network weights using prediction error.

### Q5. Why is MNIST widely used?

It is a benchmark dataset for image classification and digit recognition.

### Q6. What is a hidden layer?

A hidden layer extracts meaningful patterns from input data.

### Q7. What does Adam optimizer do?

It efficiently updates network weights during training.

### Q8. What is an epoch?

One complete pass through the training dataset.

### Q9. What does a confusion matrix show?

Correct and incorrect classifications for each class.

### Q10. What are the output classes in MNIST?

Digits 0 through 9.

