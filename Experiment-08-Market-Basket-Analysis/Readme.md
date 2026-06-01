# Experiment 8: Market Basket Analysis Using Apriori Algorithm

## Aim

To implement the Apriori Algorithm for Market Basket Analysis using the Online Retail Dataset and discover frequent itemsets and association rules among purchased products.

---

## Software Required

* Python 3.x
* Pandas
* Matplotlib
* mlxtend

---

## Libraries Used

```python
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
```

---

## Theory

### What is Market Basket Analysis?

Market Basket Analysis (MBA) is a data mining technique used to identify relationships among products purchased together by customers.

It helps businesses discover:

* Frequently purchased products
* Product associations
* Customer buying patterns
* Product recommendation opportunities

### What is the Apriori Algorithm?

Apriori is a frequent itemset mining algorithm that identifies item combinations that appear frequently in transaction data.

The algorithm follows the principle:

> If an itemset is frequent, all of its subsets must also be frequent.

Apriori is widely used for:

* Product recommendations
* Cross-selling strategies
* Retail analytics
* Customer behavior analysis

---

## Dataset Description

The Online Retail Dataset contains transactional information from an online retail store.

### Features

| Feature     | Description         |
| ----------- | ------------------- |
| InvoiceNo   | Transaction ID      |
| StockCode   | Product Code        |
| Description | Product Description |
| Quantity    | Quantity Purchased  |
| InvoiceDate | Date of Purchase    |
| UnitPrice   | Product Price       |
| CustomerID  | Customer Identifier |
| Country     | Customer Country    |

---

## Objectives

1. Load and explore the Online Retail Dataset.
2. Clean transaction data.
3. Create a basket matrix.
4. Apply binary encoding.
5. Generate frequent itemsets using Apriori.
6. Generate association rules.
7. Evaluate rules using Support, Confidence, and Lift.
8. Visualize frequently purchased products.
9. Save generated association rules.

---

## Methodology

### Step 1: Load Dataset

Read the Online Retail dataset from an Excel file.

```python
df = pd.read_excel("Online Retail.xlsx")
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

Check missing values.

```python
print(df.isnull().sum())
```

Purpose:

* Understand dataset structure.
* Identify missing values.
* Examine available features.

---

## Data Cleaning

Data cleaning is essential before applying the Apriori Algorithm.

### Remove Missing Customer IDs

Rows without Customer IDs are removed.

```python
df.dropna(
    subset=['CustomerID'],
    inplace=True
)
```

Reason:

Transactions without customer information may affect analysis quality.

---

### Remove Cancelled Transactions

Invoices beginning with "C" indicate cancellations.

```python
df = df[
    ~df['InvoiceNo']
    .astype(str)
    .str.contains('C')
]
```

Reason:

Cancelled orders do not represent actual purchases.

---

### Remove Negative Quantities

Keep only positive quantities.

```python
df = df[df['Quantity'] > 0]
```

Reason:

Negative values usually represent returns or corrections.

---

## Step 3: Create Basket Matrix

A basket matrix represents customer transactions.

### Structure

| Rows    | Transactions       |
| ------- | ------------------ |
| Columns | Products           |
| Values  | Quantity Purchased |

Create transaction matrix:

```python
basket = (
    df[df['Country'] == 'United Kingdom']
    .groupby(
        ['InvoiceNo','Description']
    )['Quantity']
    .sum()
    .unstack()
    .fillna(0)
)
```

Purpose:

Transform transaction data into a format suitable for association rule mining.

---

## Step 4: Binary Encoding

The Apriori Algorithm requires binary values.

### Encoding Rules

| Quantity | Encoded Value |
| -------- | ------------- |
| 0        | 0             |
| ≥1       | 1             |

Define encoding function:

```python
def encode_units(x):

    if x <= 0:
        return 0

    if x >= 1:
        return 1
```

Apply encoding:

```python
basket_sets = basket.applymap(
    encode_units
)
```

---

## Step 5: Apply Apriori Algorithm

Generate frequent itemsets.

```python
frequent_itemsets = apriori(
    basket_sets,
    min_support=0.02,
    use_colnames=True
)
```

Display results:

```python
print(
    frequent_itemsets.head()
)
```

---

## Frequent Itemsets

Frequent itemsets represent product combinations that appear together frequently in transactions.

Example:

```text
Bread
Butter
Bread + Butter
```

These combinations satisfy the minimum support threshold.

---

## Support

### Definition

Support measures how often an itemset appears in all transactions.

Formula:

Support(A) = Number of Transactions Containing A / Total Transactions

Example:

If Bread appears in 50 out of 1000 transactions:

Support(Bread) = 50/1000 = 0.05

Interpretation:

Bread appears in 5% of all transactions.

---

## Step 6: Generate Association Rules

Association rules identify relationships among products.

Example:

```text
Bread → Butter
```

Meaning:

Customers purchasing Bread are likely to purchase Butter.

Generate rules:

```python
rules = association_rules(
    frequent_itemsets,
    metric='lift',
    min_threshold=1
)
```

---

### Select Important Metrics

```python
rules = rules[
    [
        'antecedents',
        'consequents',
        'support',
        'confidence',
        'lift'
    ]
]
```

---

## Confidence

### Definition

Confidence measures the probability of purchasing Product B when Product A is purchased.

Formula:

Confidence(A → B)

= Transactions Containing A and B

÷ Transactions Containing A

Example:

* Bread buyers = 40
* Bread and Butter buyers = 30

Confidence = 30 / 40 = 0.75

Interpretation:

75% of Bread buyers also purchased Butter.

---

## Lift

### Definition

Lift measures the strength of association between products.

Formula:

Lift(A → B)

= Confidence(A → B)

÷ Support(B)

### Interpretation

| Lift Value | Meaning              |
| ---------- | -------------------- |
| > 1        | Positive Association |
| = 1        | No Association       |
| < 1        | Negative Association |

Higher Lift indicates stronger relationships.

---

## Step 7: Sort Rules

Display strongest rules.

```python
rules = rules.sort_values(
    by='confidence',
    ascending=False
)
```

Purpose:

Identify highly reliable product recommendations.

---

## Step 8: Rule Interpretation

Interpret top association rules.

```python
for index, row in rules.head(10).iterrows():
```

Example Output:

```text
If customer buys:
['Bread']

Then customer may also buy:
['Butter']

Support: 0.04
Confidence: 0.75
Lift: 2.10
```

---

## Visualization

### Top Selling Products

Identify most frequently purchased products.

```python
top_products = (
    df['Description']
    .value_counts()
    .head(10)
)
```

Display:

```python
print(top_products)
```

---

### Bar Chart Visualization

```python
plt.figure(figsize=(12,6))

top_products.plot(
    kind='bar'
)

plt.title(
    "Top 10 Frequently Purchased Products"
)

plt.xlabel("Products")
plt.ylabel("Frequency")

plt.show()
```

Purpose:

* Identify best-selling products.
* Understand customer preferences.

---

## Save Output

Store generated association rules.

```python
rules.to_csv(
    "association_rules_output.csv",
    index=False
)
```

Output file:

```text
association_rules_output.csv
```

---

## Results

* Online Retail dataset was successfully loaded and cleaned.
* Missing values and cancelled transactions were removed.
* Basket matrix was generated successfully.
* Binary encoding transformed transactions into Apriori-compatible format.
* Frequent itemsets were discovered using the Apriori Algorithm.
* Association rules were generated successfully.
* Support, Confidence, and Lift metrics were calculated.
* Top-selling products were visualized.
* Association rules were exported to a CSV file.

---

## Conclusion

The Apriori Algorithm was successfully implemented for Market Basket Analysis using the Online Retail Dataset. Frequent itemsets and association rules were generated to uncover hidden relationships among products. The experiment demonstrated how data mining techniques can be used to analyze customer purchasing behavior and generate meaningful product recommendations.

---

## Outcome

A Market Basket Analysis system was developed using the Apriori Algorithm. The model successfully identified frequently purchased products and generated association rules that can assist businesses in improving recommendations, increasing sales, and understanding customer buying patterns.

---

## Real-World Applications

* Amazon Product Recommendations
* Grocery Store Product Placement
* E-Commerce Recommendation Systems
* Pharmacy Product Analysis
* Restaurant Combo Suggestions
* Supermarket Layout Optimization
* Online Shopping Platforms

---

## Viva Questions

### Q1. What is Market Basket Analysis?

Market Basket Analysis is a data mining technique used to discover relationships among products purchased together.

### Q2. What is the Apriori Algorithm?

Apriori is a frequent itemset mining algorithm used to identify associations among items in transaction data.

### Q3. What is Support?

Support measures how frequently an item or itemset appears in the dataset.

### Q4. What is Confidence?

Confidence measures the probability of purchasing one item when another item is purchased.

### Q5. What is Lift?

Lift measures the strength of association between two items.

### Q6. Why is binary encoding required?

The Apriori Algorithm works on binary transaction data where purchased items are represented as 1 and non-purchased items as 0.

### Q7. Why are cancelled transactions removed?

Cancelled transactions do not represent actual purchases and can distort association patterns.

### Q8. Give one real-world application of Apriori.

Amazon's product recommendation system is a common application of Apriori-based association rule mining.

