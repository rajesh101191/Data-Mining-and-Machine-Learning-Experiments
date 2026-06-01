# Experiment 2: Classification Tree Using Loan Dataset

## Objective

To implement a Decision Tree Classification algorithm using the Loan Dataset and predict whether a customer is likely to default on a loan.

---

## Description

Decision Trees are supervised machine learning algorithms used for classification and prediction tasks. They partition data into smaller subsets based on feature values, creating a tree-like structure of decisions and outcomes.

In this experiment, a Loan Dataset is analyzed to classify customers into default and non-default categories based on their financial attributes.

---

## Dataset

The dataset contains information related to loan applicants and includes various attributes such as income, loan amount, and other financial indicators.

### Target Variable

* **Default**

  * Yes: Customer defaulted on the loan
  * No: Customer did not default on the loan

---

## Methodology

### Step 1: Data Exploration

* Load the dataset using Pandas.
* Examine dataset structure and summary statistics.
* Visualize data using boxplots and scatter plots.

### Step 2: Data Preprocessing

* Separate input features and target variable.
* Apply one-hot encoding to categorical variables.

### Step 3: Train-Test Split

* Split the dataset into training and testing subsets.
* Use stratified sampling to preserve class distribution.

### Step 4: Model Training

* Train a Decision Tree Classifier using the training data.

### Step 5: Model Evaluation

* Compute training and testing accuracy.
* Analyze model performance.

### Step 6: Tree Visualization

* Visualize the generated decision tree.
* Interpret decision rules and splitting criteria.

### Step 7: Feature Importance Analysis

* Identify the most influential features contributing to classification.

### Step 8: Hyperparameter Tuning

* Optimize model parameters using GridSearchCV.
* Improve model generalization and reduce overfitting.

---

## Libraries Used

* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn

---

## Output

The experiment generates:

* Exploratory Data Analysis Visualizations
* Decision Tree Visualization
* Feature Importance Plot
* Training Accuracy
* Testing Accuracy
* Optimized Decision Tree Model

---

## Learning Outcomes

After completing this experiment, students will be able to:

* Understand Decision Tree Classification.
* Perform exploratory data analysis.
* Preprocess categorical and numerical data.
* Train and evaluate classification models.
* Visualize decision trees.
* Interpret feature importance.
* Apply hyperparameter tuning techniques.

---

## Conclusion

The Decision Tree Classifier successfully classifies loan applicants based on their financial characteristics. The experiment demonstrates the complete workflow of a classification problem, including data preprocessing, model training, evaluation, visualization, and optimization.

