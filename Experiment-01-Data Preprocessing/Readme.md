# Experiment 1: Data Preprocessing

## Aim

To perform data preprocessing on the State Power Position dataset by handling missing values, converting data types, detecting and removing outliers, normalizing data, and preparing the dataset for further analysis.

## Software Required

* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## Dataset Description

The dataset contains information related to power demand and supply across different states and regions of India. The attributes include:

* Region
* State
* Peak Demand
* Peak Met
* Energy Requirement
* Energy Supplied
* Shortage
* Maximum Outage
* Date

## Objectives

1. Load and inspect the dataset.
2. Handle missing values.
3. Convert columns to appropriate data types.
4. Extract useful features from date attributes.
5. Detect and remove outliers.
6. Normalize numerical data.
7. Perform correlation analysis.
8. Apply Principal Component Analysis (PCA).
9. Visualize the processed data.
10. Save the cleaned dataset.

## Methodology

### Step 1: Data Loading

The dataset is loaded using Pandas and initial inspection is performed using:

* head()
* shape()
* info()

### Step 2: Data Cleaning

* Renamed columns for better readability.
* Missing values in the Region column were filled using Forward Fill (ffill).
* Missing values in numerical columns were replaced with their respective mean values.

### Step 3: Data Type Conversion

Numerical attributes were converted into numeric format using:

```python
pd.to_numeric(errors='coerce')
```

The Date column was converted into datetime format using:

```python
pd.to_datetime()
```

### Step 4: Feature Engineering

Additional features were extracted from the Date column:

* Year
* Month

### Step 5: Outlier Detection and Removal

The Interquartile Range (IQR) method was used:

* Q1 = 25th Percentile
* Q3 = 75th Percentile
* IQR = Q3 − Q1

Outliers were identified using:

```python
Lower Bound = Q1 − 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Records outside these bounds were removed.

### Step 6: Data Normalization

Peak Demand values were normalized using Min-Max Scaling:

```python
MinMaxScaler()
```

The normalized values were stored in a new column named:

* Demand_Normalized

### Step 7: Correlation Analysis

Correlation among numerical features was calculated using:

```python
df.corr()
```

This helps identify relationships between power demand and supply parameters.

### Step 8: Principal Component Analysis (PCA)

PCA was applied to reduce dimensionality while preserving maximum variance.

Steps:

1. Standardize features using StandardScaler.
2. Apply PCA.
3. Extract principal components.

### Step 9: Data Visualization

The following visualizations were generated:

1. Scree Plot

   * Shows explained variance by principal components.

2. Peak Demand Over Time

   * Line graph showing demand trends.

3. Average Demand by State

   * Bar chart comparing state-wise demand.

4. Box Plot

   * Used for visualizing outliers.

### Step 10: Save Cleaned Dataset

The processed dataset was saved as:

```text
cleaned_power_dataset.csv
```

## Results

* Missing values were successfully handled.
* Data types were standardized.
* Outliers were identified and removed.
* Numerical values were normalized.
* PCA reduced dimensionality and revealed important variance patterns.
* Visualizations provided insights into demand trends and state-wise power consumption.
* A cleaned dataset was generated for further analysis.

## Conclusion

Data preprocessing is a crucial step in data mining and machine learning. In this experiment, the State Power Position dataset was cleaned, transformed, normalized, and analyzed. The resulting dataset is free from major inconsistencies and is suitable for advanced analytical and predictive modeling tasks.

## Outcome

A clean and structured power consumption dataset was obtained, ready for exploratory data analysis, machine learning, and predictive modeling applications.

