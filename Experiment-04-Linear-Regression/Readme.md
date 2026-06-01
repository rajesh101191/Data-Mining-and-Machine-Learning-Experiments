# Experiment 5: Linear Regression Using Housing Dataset

## Aim

To implement the Linear Regression algorithm using the Housing dataset for predicting house prices and evaluate model performance using various regression metrics.

## Software Required

* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Seaborn

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
```

## Dataset Description

The Housing dataset contains information about residential properties and their selling prices.

### Attributes

* Area
* Bedrooms
* Bathrooms
* Stories
* Parking
* Main Road Access
* Guest Room
* Basement
* Hot Water Heating
* Air Conditioning
* Preferred Area
* Furnishing Status
* Price (Target Variable)

## Objectives

1. Load and explore the Housing dataset.
2. Perform data preprocessing.
3. Convert categorical variables into numerical format.
4. Split the dataset into training and testing sets.
5. Train a Linear Regression model.
6. Predict house prices using unseen data.
7. Evaluate model performance using regression metrics.
8. Analyze residual errors.
9. Identify important features affecting house prices.
10. Visualize prediction performance and feature importance.

---

## Methodology

### Step 1: Data Loading

The Housing dataset is loaded using Pandas.

```python
df = pd.read_csv("Housing.csv")
```

The first few records are displayed using:

```python
df.head()
```

---

### Step 2: Dataset Exploration

Basic information about the dataset is obtained using:

```python
df.shape
df.columns
df.info()
```

This helps understand:

* Number of rows and columns
* Data types
* Missing values
* Feature names

---

### Step 3: Data Preprocessing

Machine learning algorithms require numerical inputs.

Categorical attributes are converted into numerical format using One-Hot Encoding:

```python
df = pd.get_dummies(df, drop_first=True)
```

Benefits:

* Removes categorical dependencies.
* Prevents multicollinearity.
* Makes data suitable for Linear Regression.

---

### Step 4: Define Features and Target Variable

#### Target Variable

```python
y = df['price']
```

#### Feature Variables

```python
X = df.drop('price', axis=1)
```

Where:

* X → Independent Variables
* y → Dependent Variable (House Price)

---

### Step 5: Train-Test Split

The dataset is divided into:

* 80% Training Data
* 20% Testing Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Purpose:

* Training data is used to learn patterns.
* Testing data is used to evaluate performance.

---

### Step 6: Create Linear Regression Model

```python
model = LinearRegression()
```

Linear Regression establishes a relationship between independent variables and the target variable.

---

### Step 7: Train the Model

The model learns from training data.

```python
model.fit(X_train, y_train)
```

---

### Step 8: Model Parameters

#### Intercept

```python
print(model.intercept_)
```

The intercept represents the predicted house price when all feature values are zero.

#### Coefficients

```python
coeff_df = pd.DataFrame(
    model.coef_,
    X.columns,
    columns=['Coefficient']
)
```

Coefficients indicate the influence of each feature on house prices.

---

### Step 9: House Price Prediction

Predict prices for testing data.

```python
y_pred = model.predict(X_test)
```

Example:

```python
print(y_pred[:10])
```

---

## Model Evaluation

The following performance metrics are calculated:

### 1. Mean Absolute Error (MAE)

Measures average prediction error.

```python
mae = mean_absolute_error(y_test, y_pred)
```

Lower MAE indicates better performance.

---

### 2. Mean Squared Error (MSE)

Measures average squared error.

```python
mse = mean_squared_error(y_test, y_pred)
```

Lower MSE indicates better model accuracy.

---

### 3. Root Mean Squared Error (RMSE)

Square root of MSE.

```python
rmse = np.sqrt(mse)
```

RMSE is expressed in the same unit as the target variable.

---

### 4. R² Score

Measures goodness of fit.

```python
r2 = r2_score(y_test, y_pred)
```

Interpretation:

* R² = 1 → Perfect prediction
* R² = 0 → No predictive capability

Higher values indicate better performance.

---

## Actual vs Predicted Comparison

A comparison table is created:

```python
comparison = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})
```

This helps observe prediction accuracy.

---

## Visualization

### 1. Actual vs Predicted Prices

Scatter plot comparing actual and predicted values.

```python
plt.scatter(y_test, y_pred)
```

Observation:

Points closer to the diagonal line indicate better predictions.

---

### 2. Residual Analysis

Residuals represent prediction errors.

```python
residuals = y_test - y_pred
```

---

### 3. Residual Distribution Plot

Histogram of residuals.

```python
plt.hist(residuals)
```

Observation:

A well-performing model produces residuals centered around zero and approximately normally distributed.

---

### 4. Residuals vs Predicted Values

```python
plt.scatter(y_pred, residuals)
```

Observation:

Residuals should be randomly distributed around zero without visible patterns.

---

### 5. Feature Importance Analysis

Feature importance is determined using model coefficients.

```python
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
```

Higher coefficient magnitude indicates greater influence on house price prediction.

---

### 6. Feature Importance Visualization

```python
plt.barh(
    coeff_df['Feature'],
    coeff_df['Coefficient']
)
```

This graph helps identify the most influential housing features.

---

### 7. Correlation Matrix

Heatmap showing relationships among variables.

```python
sns.heatmap(df.corr())
```

Benefits:

* Identifies highly correlated features.
* Detects multicollinearity.
* Assists feature selection.

---

## Predicting Price for a New House

Example prediction:

```python
sample_house = pd.DataFrame({...})
predicted_price = model.predict(sample_house)
```

This demonstrates how the trained model can estimate prices for new properties.

---

## Output Generation

Prediction results are saved into a CSV file.

```python
comparison.to_csv(
    "housing_prediction_output.csv",
    index=False
)
```

---

## Results

* Housing data was successfully preprocessed.
* Categorical variables were encoded.
* Linear Regression model was trained successfully.
* House prices were predicted accurately.
* Model performance was evaluated using MAE, MSE, RMSE, and R² Score.
* Residual analysis confirmed model behavior.
* Feature importance analysis identified key factors affecting house prices.

---

## Conclusion

Linear Regression was successfully implemented on the Housing dataset for house price prediction. The model demonstrated the ability to learn relationships between housing attributes and prices. Evaluation metrics and residual analysis provided insights into prediction accuracy and model performance.

---

## Outcome

A predictive Linear Regression model was developed that can estimate house prices based on housing features. The experiment demonstrated the complete machine learning workflow including preprocessing, training, evaluation, visualization, and prediction.

