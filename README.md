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

## 3. Key Learnings

* Use **IQR** or **Z-score** methods to detect outliers.
* Always decide whether to **remove or keep** outliers based on context.
* Data cleaning rules (drop, impute, convert) must depend on conditions.
* EDA helps uncover patterns that raw numbers cannot show.

---

**Revision Tip:**

* If dataset is **skewed** → prefer **IQR method** (robust against skew).
* If dataset is **normally distributed** → use **Z-score method**.


