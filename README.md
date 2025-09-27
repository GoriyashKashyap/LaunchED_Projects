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
  Contains health, demographic, and economic variables related to life expectancy.

---

## 2. Concepts Covered

### a. Data Cleaning

* Checking for **missing values** (`isnull()`, `sum()`).
* Handling missing data:

  * Drop rows/columns (if many values are missing).
  * Impute with **mean, median, or mode**.
* Detecting **duplicates** and removing them.
* Checking **data types** and converting when needed (e.g., `object → numeric`).

### b. Descriptive Statistics

* `.describe()` for mean, median, min, max, standard deviation.
* **Shape** of dataset (`rows × columns`).
* Count of unique values.

### c. Univariate Analysis

* Distribution plots (`histogram`, `distplot`, `boxplot`).
* Detecting **outliers** (via boxplots or IQR method).

### d. Bivariate Analysis

* **Correlation matrix** (`df.corr()`).
* Heatmap to visualize correlations.
* Scatterplots between independent variables and target (`Life expectancy`).

### e. Data Visualization

* Libraries: `matplotlib`, `seaborn`.
* Plots used:

  * Histogram → distribution
  * Barplot → categorical comparison
  * Scatterplot → relationships
  * Heatmap → correlation

---

## 3. Key Learnings

* How to clean and preprocess datasets before modeling.
* Importance of detecting missing data and outliers.
* Use of statistical summaries and plots to **understand patterns**.
* Correlation helps in feature selection (choosing strong predictors).

---

## 4. Steps to Run

1. Open `EDA_Project.ipynb`.
2. Run all cells sequentially.
3. Study the summary stats and plots for insights.

---

**Revision Tip:**
When revising, focus on **why** each EDA step is done:

* Cleaning ensures valid inputs.
* Visualization reveals trends/outliers.
* Correlation helps reduce irrelevant features.


