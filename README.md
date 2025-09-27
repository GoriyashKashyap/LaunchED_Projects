# LaunchED_Projects

This repository contains several minor machine learning / data analysis projects I’ve completed for revision and demonstration.

## Projects Overview

| Project | Description | Input / Dataset(s) | Models / Techniques Used |
|---|---|---|---|
| EDA Project | Exploratory Data Analysis and insights on a dataset | `dirty_cafe_sales.csv` | Data cleaning, descriptive stats, visualization |
| Linear Regression Project | Build regression models to predict continuous target(s) | `Life Expectancy Data.csv` | Linear regression, model evaluation (RMSE, \(R^2\Z_Score), etc.) |
| Logistic Regression Project | Classification / binary prediction | `smoking.csv` | Logistic regression, confusion matrix, accuracy and related metrics |

## File / Directory Structure

├── EDA_Project.ipynb \
├── LinearRegression_Project.ipynb\
├── Logistic_Regression.ipynb\
├── Life Expectancy Data.csv\
├── dirty_cafe_sales.csv\
├── smoking.csv\
└── README.md


# Exploratory Data Analysis (EDA) Project

This project focuses on performing **Exploratory Data Analysis (EDA)** to understand the dataset, detect patterns, handle missing values, and visualize relationships.

---

## 1. Dataset Used

* `dirty_cafe_sales.csv`
  Contains raw transactional sales data from a cafe, with errors, missing values, and inconsistencies.

---

## 2. Concepts Covered

### a. Data Cleaning

The dataset contains issues like missing values, duplicates, wrong data types, and outliers. Cleaning steps include:

1. **Handling Missing Values**

   * If only a **few rows** are missing → drop them using `dropna()`.
   * If a **column has many missing values** → drop the entire column.
   * If missing values are important but small → impute with:

     * **Mean/Median** for numerical values (e.g., sales amount, price) :
         * **Mean** : If the column don't have any outlier use Mean.
         * **Medain** : If the column have outlier use Mean.
     * **Mode** for categorical values (e.g., product name, category).

2. **Removing Duplicates**

   * Checked with `df.duplicated()`.
   * Removed using `drop_duplicates()` to avoid double-counting sales.

3. **Data Type Conversion**

   * Converted **date columns** into `datetime`.
   * Ensured **numeric columns** (like price, quantity, total) are integers/floats.
   * Categorical values (e.g., item type) converted to `category` dtype.

#### 4. Handling Outliers

Outliers are data points that deviate significantly from the rest.
Two common approaches:

**(i) IQR Method (Interquartile Range)**

1. Calculate Q1 (25th percentile) and Q3 (75th percentile).
2. Compute **IQR = Q3 – Q1**.
3. Define lower and upper bounds:

$$
\text{Lower Bound} = Q1 - 1.5 \times IQR
$$

$$
\text{Upper Bound} = Q3 + 1.5 \times IQR
$$

4. Any data point outside these bounds is treated as an outlier.

**Example:** If `quantity` has Q1 = 2 and Q3 = 5 → IQR = 3.

* Lower Bound = 2 – (1.5 × 3) = –2.5 → practically 0 (since quantity can’t be negative).
* Upper Bound = 5 + (1.5 × 3) = 9.5.
  → So, any `quantity > 9.5` is an outlier.

**Decision Rule:**

* If the outlier is a **data entry mistake** (e.g., quantity = 1000), remove it.
* If the outlier is **real but rare** (e.g., group order of 12 coffees), keep it.

---

**(ii) Z-Score Method**

1. Compute the **Z-score** for each data point:

$$
Z = \frac{(x - \mu)}{\sigma}
$$

Where:

* $x$ = data point
* $\mu$ = mean
* $\sigma$ = standard deviation

2. If $|Z| > 3$, the point is usually considered an outlier.



### b. Descriptive Statistics

* `.describe()` for mean, median, min, max, standard deviation.
* **Shape** of dataset (`rows × columns`).
* Count of unique values.
---

### d. Bivariate Analysis

* Scatterplot: Quantity vs. Total Sales.
* Correlation heatmap.
* Grouped sales by day/category.

---

### e. Visualization

* **Matplotlib** and **Seaborn** used for:

  * Top-selling items
  * Daily sales trends
  * Outlier detection via boxplots

---
## Data Preprocessing: Normalization, Scaling, and Encoding

Before feeding data into machine learning models, it is important to preprocess it so that features are on a similar scale and categorical data is properly encoded.

---

### 1. Normalization
Normalization rescales numeric features to a **common range**, usually [0, 1], without distorting differences in the range of values.

**Formula:**
  X'=   $$\frac{X - X_{min}}{X_{max} - X_{min}} $$

- **When to use:** When features have different units or ranges.  
- Prevents features with large values from dominating smaller-valued features.

**Python Example:**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

```
---
### 2.Scaling (Standardization)

Scaling adjusts numeric features so they have **mean = 0** and **standard deviation = 1**, which helps models that are sensitive to feature magnitude.

**Formula (Z-score):**

Z = $$\frac{X - \mu}{\sigma}$$

Where:  
-  $$X$$  = original feature value  
-  $$\mu $$ = mean of the feature  
-  $$\sigma$$  = standard deviation of the feature  

**When to Use:**  
- For models like Linear Regression, Logistic Regression, SVM, or PCA.  
- When features have different scales or units.  

**Python Example:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
---
### 3.Encoding (One-Hot Encoding)

Categorical variables must be converted into numeric format so machine learning models can process them.  
**One-Hot Encoding** creates a binary column for each category.

**Example:**  
- Column: `Status` = {Developed, Developing}  
- After One-Hot Encoding:  
  - Developed → [1, 0]  
  - Developing → [0, 1]  

**Python Example:**
```python
import pandas as pd

# One-hot encode the 'Status' column
df = pd.get_dummies(df, columns=['Status'], drop_first=True)
```
* `drop_first=True` : avoids the dummy variable trap (multicollinearity between columns). 
* **When to Use** :
    * Any categorical variable (nominal or ordinal) that cannot be interpreted as numeric by the model.
    * Ensures each category is represented separately without implying ordinal relationships.
---

## 3. Key Learnings

* Use **IQR** or **Z-score** methods to detect outliers.
* Always decide whether to **remove or keep** outliers based on context.
* Data cleaning rules (drop, impute, convert) must depend on conditions.
* EDA helps uncover patterns that raw numbers cannot show.

---

**Revision Tip:**

* If dataset is **skewed** → prefer **IQR method** (robust against skew).
* If dataset is **normally distributed** → use **Z-score method**.

---







# Linear Regression Project 

This project applies **Linear Regression** to analyze and predict **life expectancy** based on various socio-economic and health-related factors.

It demonstrates **data cleaning, regression modeling, evaluation, and interpretation** in detail.

---

## 1. Dataset Used

* **`Life Expectancy Data.csv`**
  Contains data on life expectancy across different countries, with predictors such as:

  * GDP, schooling, health expenditure
  * Adult mortality, infant deaths
  * Alcohol consumption, BMI, immunization coverage
  * Other socio-economic and health indicators

**Goal:** Predict the **Life Expectancy (in years)** using regression techniques.

---

## 2. Concepts & Steps Covered

### a. What is Linear Regression?

Linear regression models the relationship between a dependent variable $y$ (Life Expectancy) and independent variables $x_1, x_2, \ldots, x_n$ (predictors).

**Equation:**

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n + \epsilon$$

Where:

* $y$ = Life Expectancy
* $x_i$ = Independent features (e.g., GDP, schooling, BMI)
* $\beta_0$ = Intercept (baseline value when all $x_i=0$)
* $\beta_i$ = Coefficients showing how much life expectancy changes when $x_i$ increases by 1 unit (keeping others constant)
* $\epsilon$ = Error term

---

### b. Types of Linear Regression

1. **Simple Linear Regression**

   * One predictor, one target.
   * Equation:

   $$y = \beta_0 + \beta_1 x$$

2. **Multiple Linear Regression**

   * Multiple predictors.
   * Equation:

   $$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n$$

In this project, **Multiple Linear Regression** is used (since life expectancy depends on several factors).

---

### c. Assumptions of Linear Regression

Linear regression relies on some **statistical assumptions**:

1. **Linearity** → Relationship between predictors and target is linear.

   * Checked with scatterplots (GDP vs Life Expectancy).

2. **Independence of Errors** → Residuals are independent.

   * Checked with **Durbin-Watson test**.

3. **Homoscedasticity** → Residuals have constant variance.

   * Checked with residual plots.

4. **Normality of Residuals** → Errors are normally distributed.

   * Checked with histogram or Q-Q plot of residuals.

5. **No Multicollinearity** → Predictors are not highly correlated with each other.

   * Checked using **Variance Inflation Factor (VIF)**.

---

### d. Cost Function (Ordinary Least Squares)

The model tries to minimize the **Sum of Squared Errors (SSE)** between actual and predicted life expectancy.

$$
J(\beta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:

* $y_i$ = actual life expectancy
* $\hat{y}_i$ = predicted life expectancy
* $n$ = number of data points

This is known as the **Mean Squared Error (MSE)**.

---

### e. Optimization – Gradient Descent

The model parameters ($\beta$) are adjusted iteratively to minimize cost:

$$
\beta_j := \beta_j - \alpha \cdot \frac{\partial J}{\partial \beta_j}
$$

Where:

* $\alpha$ = learning rate
* $\frac{\partial J}{\partial \beta_j}$ = derivative of cost w.r.t coefficient

---
## f. Train-Test Split

To evaluate the performance of our model, we split the dataset into **training** and **testing** subsets.

- **Training Set** → Used to fit the model (learn the parameters).
- **Testing Set** → Used to evaluate the model on unseen data.

### Formula:
If dataset = \(D \), with \( n \) samples:  

- Training set size = $$\( \alpha \times n \) (commonly \( \alpha = 0.7 \))  $$
- Testing set size = $$\( (1 - \alpha) \times n \)  $$

Example in Python:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
---
### g. Fitting the Linear Regression Model

Linear Regression fits a line (or hyperplane in multiple dimensions) to predict a continuous target variable from one or more features.

**Equation:** 
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n + \epsilon$$

Where:
- **$y$** = predicted target (e.g., Life Expectancy)
- **$x_i$** = features/predictors (e.g., GDP, schooling, BMI)
- **$\beta_0$** = intercept
- **$\beta_i$** = coefficients (weights)
- **$\epsilon$** = error term
  
**Steps to Fit in Python:**
```python
from sklearn.linear_model import LinearRegression

# Create model object
lr = LinearRegression()

# Fit the model on training data
lr.fit(X_train, y_train)

# Make predictions on test data
y_pred = lr.predict(X_test)
```
---

### h. Model Evaluation Metrics

1. **R-squared (Coefficient of Determination):**

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

* Explains how much variation in life expectancy is explained by the model.
* Example: $R^2 = 0.85$ → 85% of variation is explained.

---

2. **Adjusted R-squared**
   Penalizes adding useless predictors.

$$
R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}
$$

Where:

* $n$ = number of observations
* $p$ = number of predictors

---

3. **Root Mean Squared Error (RMSE):**

$$
RMSE = \sqrt{ \frac{1}{n} \sum (y_i - \hat{y}_i)^2 }
$$

* Average size of prediction error in years.

---

4. **Mean Absolute Error (MAE):**

$$
MAE = \frac{1}{n} \sum |y_i - \hat{y}_i|
$$

* Easier interpretation (e.g., model is off by ~2 years on average).

---

### i. Residual Analysis

Residuals = $y - \hat{y}$

* **Random scatter** → good fit.
* **Patterns** → model assumptions violated.
* **Large residuals** → possible outliers.

---

### j. Feature Importance (Coefficients)

* Positive coefficient ($\beta > 0$) → Increases life expectancy.

  * Example: More **schooling years** increases life expectancy.
* Negative coefficient ($\beta < 0$) → Decreases life expectancy.

  * Example: Higher **adult mortality** decreases life expectancy.

---

## 3. Key Learnings

* Life expectancy strongly correlates with socio-economic and health features.
* **Linear regression** explains relationships and gives interpretable coefficients.
* Evaluation metrics (R², RMSE, MAE) measure goodness of fit.
* Checking assumptions ensures reliable results.

---

## 4. Practical Steps in Notebook

1. Load `Life Expectancy Data.csv`.
2. Clean missing values (drop/impute).
3. Split into features (GDP, schooling, etc.) and target (Life Expectancy).
4. Train-test split.
5. Fit **Linear Regression model**.
6. Evaluate using R², Adjusted R², RMSE, MAE.
7. Interpret coefficients and residuals.

---

**Revision Tip:**

* Write the **regression line equation** from memory.
* Recall **cost function** (MSE).
* Know **R² vs Adjusted R²** difference.
* Always check assumptions before trusting results.

