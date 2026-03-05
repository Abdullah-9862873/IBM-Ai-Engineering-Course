# Introduction to Regression

## What is Regression?
-> Type of **supervised learning** model
-> Models relationship between a **continuous target variable** and **explanatory features**
-> Predicts a continuous value (like CO2 emissions, price, temperature)

### Example: CO2 Emissions
-> Dataset: Car features (engine size, cylinders, fuel consumption)
-> Goal: Predict CO2 emission from these features

---

## Types of Regression

### 1. Simple Regression
-> **Single independent variable** estimates dependent variable
-> Can be linear or nonlinear
-> Example: Predict CO2 using only engine size

### 2. Multiple Regression
-> **More than one independent variable** estimates dependent variable
-> Example: Predict CO2 using engine size AND number of cylinders

### Linear vs Nonlinear:
- **Linear**: Straight line relationship (y = mx + c)
- **Curved/Nonlinear**: Curved relationship

---

## Regression Applications

| Application | Example |
|-------------|---------|
| **Sales Forecasting** | Predict yearly sales from customers, leads, order history |
| **House Price Prediction** | Predict price from size, bedrooms, location |
| **Predictive Maintenance** | Predict when machine needs maintenance before failure |
| **Income Prediction** | Predict salary from hours, education, experience |
| **Weather Prediction** | Estimate rainfall from temperature, humidity, wind |
| **Healthcare** | Predict disease spread, likelihood of diabetes/heart disease |
| **Environmental** | Predict wildfire probability and severity |

---

## Regression Algorithms

### Classical Methods:
- Linear Regression
- Polynomial Regression

### Modern ML Methods:
- Random Forest
- XGBoost
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Neural Networks

---

## Summary:
-> Regression = Supervised learning to predict continuous values
-> Simple = 1 feature, Multiple = 2+ features
-> Can be linear (straight) or nonlinear (curved)
-> Used in finance, healthcare, retail, weather, and more
