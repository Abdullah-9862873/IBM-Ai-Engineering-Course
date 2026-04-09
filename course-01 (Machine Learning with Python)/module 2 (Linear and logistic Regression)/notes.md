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

---

# Introduction to Simple Linear Regression

## What is Simple Linear Regression?
-> Models a **linear relationship** between one independent variable and a continuous target
-> Uses a straight line: y = mx + c (or ŷ = θ₀ + θ₁x)

### Key Terms:
- **y or ŷ (y-hat)** = Predicted value (target)
- **x** = Independent variable (predictor)
- **θ₀ (theta-zero)** = Intercept (where line crosses y-axis)
- **θ₁ (theta-one)** = Slope (how much y changes when x increases by 1)

---

## Simple Linear Regression Formula

```
ŷ = θ₀ + θ₁x

Where:
- ŷ = Predicted CO2 emission
- θ₀ = Intercept (bias)
- θ₁ = Slope (coefficient)
- x = Engine size
```

---

## Example: CO2 Emissions Dataset

### Sample Data:
| Engine Size (x) | CO2 Emission (y) |
|-----------------|------------------|
| 2.0 | 116 |
| 2.4 | 129 |
| 3.1 | 175 |
| 3.2 | 170 |
| 3.5 | 190 |
| 3.6 | 198 |
| 4.0 | 221 |
| 4.2 | 263 |
| 4.8 | 273 |
| 5.4 | 275 |

---

## How to Calculate θ₁ (Slope) and θ₀ (Intercept)

### Step 1: Calculate Means
```
x̄ = Mean of engine sizes = (2.0+2.4+3.1+3.2+3.5+3.6+4.0+4.2+4.8+5.4) / 10
x̄ = 36.2 / 10 = 3.62

ȳ = Mean of CO2 emissions = (116+129+175+170+190+198+221+263+273+275) / 10
ȳ = 2010 / 10 = 201
```

### Step 2: Calculate θ₁ (Slope)
```
θ₁ = Σ[(xi - x̄)(yi - ȳ)] / Σ[(xi - x̄)²]

Let's calculate:
xi     yi    (xi-x̄)   (yi-ȳ)  (xi-x̄)(yi-ȳ)  (xi-x̄)²
2.0    116   -1.62    -85      137.7         2.6244
2.4    129   -1.22    -72      87.84         1.4884
3.1    175   -0.52    -26      13.52         0.2704
3.2    170   -0.42    -31      13.02         0.1764
3.5    190   -0.12    -11      1.32          0.0144
3.6    198   -0.02    -3       0.06          0.0004
4.0    221    0.38     20       7.6           0.1444
4.2    263    0.58     62      35.96          0.3364
4.8    273    1.18     72      84.96          1.3924
5.4    275    1.78     74      131.72         3.1684

Σ(xi-x̄)(yi-ȳ) = 513.72
Σ(xi-x̄)² = 9.6156

θ₁ = 513.72 / 9.6156 = 53.43
```

### Step 3: Calculate θ₀ (Intercept)
```
θ₀ = ȳ - θ₁x̄
θ₀ = 201 - (53.43 × 3.62)
θ₀ = 201 - 193.42 = 7.58
```

### Final Equation:
```
ŷ = 7.58 + 53.43 × Engine_Size
```

---

## Making Predictions

### Example 1: Engine Size = 2.4
```
ŷ = 7.58 + 53.43 × 2.4
ŷ = 7.58 + 128.23
ŷ = 135.81
```
Predicted CO2 emission ≈ 136

### Example 2: Engine Size = 4.0
```
ŷ = 7.58 + 53.43 × 4.0
ŷ = 7.58 + 213.72
ŷ = 221.30
```
Predicted CO2 emission ≈ 221

---

## Residual Error

### What is Residual Error?
-> Vertical distance from data point to the regression line
-> Actual value - Predicted value

### Example:
For engine size 4.8, actual CO2 = 273
```
Predicted = 7.58 + 53.43 × 4.8 = 264.24
Residual = 273 - 264.24 = 8.76
```

### Mean Squared Error (MSE):
-> Average of all squared residual errors
-> Formula: MSE = (1/n) × Σ(yi - ŷi)²
-> Goal: Minimize MSE (better fit)

---

## OLS (Ordinary Least Squares) Regression

### What is OLS?
-> Method to find the best-fit line by minimizing MSE
-> Developed by Gauss and Legendre in 1800s
-> Solution is just a calculation (no tuning needed)

### Advantages:
- Easy to understand and interpret
- No hyperparameters to tune
- Fast calculation (especially for small datasets)

### Disadvantages:
- May be too simplistic for complex data
- Cannot capture nonlinear relationships
- Outliers can greatly reduce accuracy

---

## Summary:
-> Simple Linear Regression = Straight line relationship between 1 variable and target
-> Formula: ŷ = θ₀ + θ₁x
-> Calculate θ₁ using: Σ(xi-x̄)(yi-ȳ) / Σ(xi-x̄)²
-> Calculate θ₀ using: ȳ - θ₁x̄
-> Best-fit line minimizes residual errors (MSE)
-> OLS = Ordinary Least Squares (most common method)

---

# Introduction to Multiple Linear Regression

## What is Multiple Linear Regression?
-> Extension of Simple Linear Regression
-> Uses **two or more independent variables** to estimate a dependent variable
-> Better model than simple linear regression

### Formula:
```
ŷ = θ₀ + θ₁x₁ + θ₂x₂ + θ₃x₃ + ... + θₙxₙ

Where:
- ŷ = Predicted value (target)
- θ₀ = Intercept (bias)
- θ₁, θ₂, θ₃... = Coefficients for each feature
- x₁, x₂, x₃... = Independent variables (features)
```

---

## Simple vs Multiple Linear Regression

| Simple Linear Regression | Multiple Linear Regression |
|--------------------------|---------------------------|
| 1 independent variable | 2+ independent variables |
| Formula: ŷ = θ₀ + θ₁x | Formula: ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... |
| Creates a **line** | Creates a **plane** (2 features) or **hyperplane** (3+ features) |
| Easier to interpret | More complex but more accurate |

---

## Example: CO2 Emissions Prediction

### Dataset Features:
- Engine Size (x₁)
- Number of Cylinders (x₂)
- Fuel Consumption (x₃)

### Sample Parameters:
```
θ₀ = 62.43
θ₁ = 9.19 (engine size)
θ₂ = 8.70 (cylinders)
θ₃ = ... (fuel consumption)
```

### Prediction for Record #9:
```
Car specs: Engine size = 2.4, Cylinders = 4

ŷ = 62.43 + (9.19 × 2.4) + (8.70 × 4) + ...
ŷ = 62.43 + 22.06 + 34.80 + ...
ŷ = 208.34

Predicted CO2 emission = 208.34
```

---

## Handling Categorical Variables

### Binary Categories (Yes/No):
-> Convert to 0 and 1
-> Example: Car Type
   - Manual = 0
   - Automatic = 1

### Multiple Categories:
-> Create separate Boolean features for each class
-> Example: Color (Red, Blue, Green)
   - Is_Red = 0 or 1
   - Is_Blue = 0 or 1
   - Is_Green = 0 or 1

---

## Applications of Multiple Linear Regression

| Field | Example |
|-------|---------|
| **Education** | Predict exam scores from revision time, test anxiety, lecture attendance |
| **Healthcare** | Predict blood pressure changes from BMI changes |
| **Finance** | Predict house prices from size, bedrooms, location, age |
| **Business** | Predict sales from advertising spend, price, season |
| **Sports** | Predict player performance from training hours, age, experience |

---

## What-If Scenarios

### What is What-If Analysis?
-> Change one or more input features to see predicted outcome
-> Helps understand relationship between variables

### Example:
-> "How much will blood pressure change if BMI increases by 1?"

---

## Pitfalls of Multiple Linear Regression

### 1. Overfitting
-> Adding too many variables
-> Model memorizes training data
-> Poor prediction for new data

### 2. Collinearity (Correlated Variables)
-> When two variables are correlated, they predict each other
-> Example: Engine size and fuel consumption are correlated
-> Solution: Remove redundant variables

### 3. Impossible Scenarios
-> Asking model to predict impossible situations
-> Example: Negative age, negative price

### 4. Extrapolation
-> Predicting far outside the range of training data
-> Model may give inaccurate results

---

## How to Build a Good Model

### Variable Selection:
- Use **uncorrelated** variables
- Choose variables most **understood** and **controllable**
- Pick variables most **correlated with target**

### Relative Importance:
-> Multiple regression shows how much each feature contributes
-> Helps identify most important predictors

---

## Methods to Find Best Parameters

### 1. Ordinary Least Squares (OLS)
-> Use linear algebra to calculate optimal θ values
-> Minimizes MSE mathematically

### 2. Gradient Descent (Optimization)
-> Start with random values
-> Iteratively minimize error
-> Good for large datasets

---

## Summary:
-> Multiple Linear Regression = 2+ features predicting continuous target
-> Formula: ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
-> Better than simple regression but can overfit
-> Handle categorical variables by converting to numbers
-> Pitfalls: Overfitting, Collinearity, Impossible scenarios, Extrapolation
-> Use OLS or Gradient Descent to find best parameters

---

# Introduction to Polynomial and Nonlinear Regression

## What is Nonlinear Regression?
-> Statistical method for modeling relationship between dependent and independent variables
-> Relationship is represented by a **nonlinear equation**
-> Can use polynomial, exponential, logarithmic, or other non-linear functions
-> Useful when complex relationship between variables cannot be captured through a straight line

### When to Use Nonlinear Regression:
-> When dataset follows an exponential growth pattern
-> When data has a background trend that follows a smoothed curve
-> When straight line **underfits** the data

---

## Polynomial Regression

### What is Polynomial Regression?
-> Uses ordinary linear regression to fit polynomial expressions of features
-> Relationship between independent variable x and dependent variable y is modeled as **nth degree polynomial**
-> Can be transformed into linear regression problem

### Formula (Cubic/Third Degree):
```
y = θ₀ + θ₁x + θ₂x² + θ₃x³
```

### How to Convert to Linear Regression:
-> Create new variables: x₁ = x, x₂ = x², x₃ = x³
-> Model becomes: y = θ₀ + θ₁x₁ + θ₂x₂ + θ₃x₃
-> Now use ordinary multiple linear regression to find best fit

### Polynomial Degrees:
| Degree | Name | Example |
|--------|------|---------|
| 1 | Linear | y = θ₀ + θ₁x |
| 2 | Quadratic | y = θ₀ + θ₁x + θ₂x² |
| 3 | Cubic | y = θ₀ + θ₁x + θ₂x² + θ₃x³ |

### Overfitting in Polynomial Regression:
-> High degree polynomial can pass through every point
-> Model memorizes training data including random noise
-> Result: Perfect fit on training data, poor predictions on new data
-> **Solution**: Pick a regression that fits data well without overfitting

---

## Nonlinear Regression vs Polynomial Regression

### Polynomial Regression:
-> Special form of nonlinear regression
-> Expresses nonlinear dependence on input features
-> Has **linear dependence on regression coefficients**
-> Can be transformed into linear regression problem

### Nonlinear Regression:
-> Cannot be reduced to linear regression
-> Uses functions of features (logarithm, exponential, etc.)
-> Examples: Exponential growth, logarithmic, sinusoidal

---

## Common Examples of Nonlinear Regression

### 1. Exponential/Compound Growth
-> Example: How investments grow with compound interest rates
-> Example: China's GDP growth over time (1960-2014)
-> GDP increases over time, and rate of growth also increases
-> Formula: ŷ = θ₀ + θ₁ × e^x

### 2. Logarithmic (Diminishing Returns)
-> Example: Law of diminishing returns
-> Incremental gains in productivity reduce as investment increases
-> First 6 hours show linear increase, then returns slow down logarithmically

### 3. Periodicity (Sinusoidal)
-> Example: Seasonal variations in quantity
-> Monthly rainfall or temperature patterns
-> Uses sine/cosine functions

---

## How to Determine What Kind of Regression Model

### Step 1: Visual Analysis
-> Analyze scatter plots of target variable against each input variable
-> Look for patterns in dependencies
-> Determine if relationship is linear or nonlinear

### Step 2: Express Patterns Mathematically
-> Try to express patterns as mathematical functions
-> Determine if they're linear, exponential, logarithmic, or sinusoidal

### Step 3: Generate Models and Analyze Results
-> Check if data has any relationship with target
-> Visually interpret model's errors by plotting predictions against actual values

---

## How to Find Optimal Nonlinear Model

### If You Have a Mathematical Expression:
-> Use optimization technique like **gradient descent**
-> Iteratively find optimal parameters

### If You Haven't Decided on a Specific Model:
-> Select from machine learning models:
  - Regression Trees
  - Random Forests
  - Neural Networks
  - Support Vector Machines
  - Gradient Boosting Machines
  - K-Nearest Neighbors

---

## Summary:
-> Nonlinear regression uses polynomial, exponential, logarithmic equations
-> Used when relationship cannot be captured through a straight line
-> Polynomial regression fits data to polynomial expressions
-> Overfitting occurs when model memorizes noise rather than patterns
-> Common nonlinear examples: exponential, logarithmic, periodicity
-> Use scatter plots to determine appropriate regression model
-> Use gradient descent or ML models to find optimal parameters

---

# Introduction to Logistic Regression

## What is Logistic Regression?
-> Statistical modeling technique that predicts probability of an observation belonging to one of two classes
-> In machine learning, refers to a **binary classifier** based on statistical logistic regression
-> By choosing a threshold probability, becomes a binary classifier
-> Assigns each observation to one class if probability > threshold, other class if < threshold

---

## When is Logistic Regression a Good Choice?

### 1. Binary Target Variable
-> Target in data is binary (0 or 1)
-> Example: Yes/No, Churn/Stay, Fraud/Not Fraud

### 2. Need Probability of Outcome
-> Example: Probability of customer buying a product
-> Example: Probability of disease occurrence

### 3. Linearly Separable Data
-> Decision boundary is a line, plane, or hyperplane
-> Formula: θ₀ + θ₁x₁ + θ₂x₂ > 0

### 4. Understanding Feature Impact
-> Allows selection of best features based on coefficient weights
-> Helps understand impact of independent features

---

## Logistic Regression as Binary Classifier

### Applications:
| Application | Example |
|-------------|---------|
| **Healthcare** | Predict probability of heart attack based on age, sex, BMI |
| **Medical Diagnosis** | Predict probability of diabetes based on weight, blood pressure |
| **Customer Behavior** | Predict likelihood of customer churning |
| **Finance** | Predict probability of loan default |
| **Quality Control** | Predict probability of product failure |

---

## Why Not Linear Regression for Binary Classification?

### Problem with Linear Regression:
-> Predicted values increase indefinitely with input
-> Linear regression outputs can be any number, not just 0-1
-> Cannot directly predict probability

### Example: Customer Churn Prediction
-> Using linear regression: ŷ = θ₀ + θ₁x
-> As age increases, ŷ increases without bound
-> Cannot represent probability between 0 and 1

---

## Step Function Approach

### Threshold Method:
-> Use threshold (e.g., 0.5) to differentiate classes
-> Rule: If ŷ < 0.5 → class 0, otherwise → class 1

### Problem:
-> No difference between a customer 20 years old and 100 years old
-> Both would be class 1 if above threshold
-> Doesn't provide probability of belonging to a class

---

## Sigmoid Function (Logit Function)

### Formula:
```
σ(x) = 1 / (1 + e^(-x))
```

### Properties:
| x Value | σ(x) Output |
|---------|-------------|
| x = 0 | 0.5 |
| x → ∞ | 1 |
| x → -∞ | 0 |

### Key Feature:
-> Takes any continuous function of x
-> Compresses output within range 0 to 1
-> Defines a probability

---

## Logistic Regression Model

### Model with Sigmoid:
```
p̂ = σ(ŷ) = σ(θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ)
```

### Where:
-> p̂ = Predicted probability that y = 1 given x
-> σ = Sigmoid function
-> θ₀, θ₁, θ₂... = Coefficients/weights

### Decision Boundary:
-> Set threshold (usually 0.5)
-> If p̂ > 0.5 → predict class 1
-> If p̂ ≤ 0.5 → predict class 0

---

## Probabilities

### For Binary Classification:
-> Probability of class 1: p̂ = P(y=1|x)
-> Probability of class 0: 1 - p̂ = P(y=0|x)
-> Probabilities must add to 1

### Example:
-> If P(churn) = 0.8
-> Then P(stay) = 1 - 0.8 = 0.2

---

## Example: Customer Churn Prediction

### Dataset Features:
-> Customer age
-> Income
-> Services signed up for
-> Account information
-> Demographic information

### Target Variable:
-> Churn (0 = No, 1 = Yes)

### Prediction Process:
1. Input features (age, income, etc.)
2. Calculate θ₀ + θ₁x₁ + θ₂x₂ + ...
3. Apply sigmoid function
4. Get probability p̂
5. Compare with threshold (0.5)
6. Assign class

---

## Summary:
-> Logistic regression predicts probability of binary outcome
-> Uses sigmoid function to compress values between 0 and 1
-> Good for: binary targets, probabilistic results, feature impact analysis
-> Both a probability predictor and binary classifier
-> Decision boundary: threshold (typically 0.5) determines class assignment
-> Probabilities of both classes must add to 1

---

# Training a Logistic Regression Model

## Objective of Training
-> Find the best parameters (θ) that map input features to target outcomes
-> Predict classes with minimal error
-> Find set of parameters that **minimizes the cost function**

---

## Training Process Steps

### Step 1: Initialize Parameters
-> Choose a starting set of parameters (θ)
-> Can be a random choice

### Step 2: Make Predictions
-> Predict probability that class is 1 for each observation

### Step 3: Calculate Error
-> Measure error between predicted classes and actual classes
-> Error is called the **cost function**

### Step 4: Update Parameters
-> Determine new θ that reduces prediction error

### Step 5: Repeat
-> Repeat process until:
  - Log loss is small enough, OR
  - Maximum iterations reached

---

## Preliminary vs Optimal Logistic Regression

### Preliminary Logistic Regression:
-> Created by combining linear model ŷ with sigmoid function
-> Not necessarily the best model

### Optimal Logistic Regression:
-> Achieved after optimization step
-> Requires finding best parameters θ
-> Uses optimization to minimize cost function

---

## Cost Function: Log Loss

### What is Log Loss?
-> Metric for optimizing logistic regression
-> Measures how well predicted probabilities match actual classes
-> Must be **minimized**

### Formula:
```
Log Loss = -1/n × Σ[yi × log(p̂i) + (1 - yi) × log(1 - p̂i)]

Where:
- n = number of observations
- yi = actual class (0 or 1)
- p̂i = predicted probability that class is 1
```

### How Log Loss Works:
| Scenario | Predicted | Actual | Log Loss |
|----------|-----------|--------|----------|
| Confident & Correct | p̂ ≈ 1 | y = 1 | Small (vanishes) |
| Confident & Correct | p̂ ≈ 0 | y = 0 | Small (vanishes) |
| Confident & Incorrect | p̂ ≈ 1 | y = 0 | Very Large |
| Confident & Incorrect | p̂ ≈ 0 | y = 1 | Very Large |

### Key Property:
-> **Favors** confident classifications that are correct
-> **Penalizes** confident, incorrect predictions

---

## Gradient Descent

### What is Gradient Descent?
-> Iterative approach to finding minimum of a function
-> Adjusts parameter values in direction of steepest descent
-> Uses derivative of log loss function

### Key Components:
-> **Learning Rate**: Controls how far parameters can step on each iteration
-> **Gradient**: Points in direction of steepest ascent
-> **Negative Gradient**: steepest descent

### Process:
1. Calculate gradient of cost function
2. Move in opposite Points in direction of direction of gradient
3. Scale step by learning rate
4. Repeat until convergence

### Properties:
-> Gradient calculated over **entire dataset** each iteration
-> When dataset is large, gradient descent becomes very slow
-> Increasing learning rate speeds up convergence but may miss minima

### Visualization:
-> Cost function forms a surface (like a bowl)
-> Steeper slope = greater step toward minimum
-> As lowest point is reached, slope diminishes to zero
-> Lowest point = optimal θ values

---

## Stochastic Gradient Descent (SGD)

### What is SGD?
-> Variation of gradient descent
-> Uses **random subset** of training data to calculate gradient

### Advantages:
-> Faster than standard gradient descent
-> Scales well with large datasets
-> More likely to find global minimum (less likely to get stuck in local minima)

### Disadvantages:
-> Less accurate than standard gradient descent
-> Can wander around the minimum

### Convergence Improvement:
-> Slow down as algorithm gets closer to global minimum
-> Decrease learning rate as you approach minimum
-> Gradually increase size of random data sample

### Comparison:

| Feature | Gradient Descent | Stochastic Gradient Descent |
|---------|------------------|------------------------------|
| Data Used | Entire dataset | Random subset |
| Speed | Slow for large data | Faster |
| Accuracy | More accurate | Less accurate |
| Convergence | Smoother | Can wander |
| Local Minima | May get stuck | Can escape |

---

## Summary:
-> Training seeks to find parameters θ that minimize cost function
-> Log loss is the cost function that measures prediction accuracy
-> Log loss penalizes confident, incorrect predictions
-> Gradient descent iteratively finds minimum using derivatives
-> SGD uses random subsets for faster, scalable training
-> Stop training when log loss is satisfactory or max iterations reached

---

# Module Summary

Regression models relationships between a continuous target variable and explanatory features, covering simple and multiple regression types.

Simple regression uses a single independent variable to estimate a dependent variable, while multiple regression involves more than one independent variable.

Regression is widely applicable, from forecasting sales and estimating maintenance costs to predicting rainfall and disease spread.

In simple linear regression, a best-fit line minimizes errors, measured by Mean Squared Error (MSE); this approach is known as Ordinary Least Squares (OLS).

OLS regression is easy to interpret but sensitive to outliers, which can impact accuracy.

Multiple linear regression extends simple linear regression by using multiple variables to predict outcomes and analyze variable relationships.

Adding too many variables can lead to overfitting, so careful variable selection is necessary to build a balanced model.

Nonlinear regression models complex relationships using polynomial, exponential, or logarithmic functions when data does not fit a straight line.

Polynomial regression can fit data but may overfit by capturing random noise rather than the underlying patterns.

Logistic regression is a probability predictor and binary classifier, suitable for binary targets and assessing feature impact.

Logistic regression minimizes errors using log-loss and optimizes with gradient descent or stochastic gradient descent for efficiency.

Gradient descent is an iterative process to minimize the cost function, which is crucial for training logistic regression models.

