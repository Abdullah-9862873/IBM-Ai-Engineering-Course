# Quiz Answers

## Question 1
Which of the following regression methods is a modern machine learning technique?

- Linear regression
- Simple linear regression
- **Random forest regression** ✓
- Polynomial regression

**Answer:** Random forest regression

**Reason:** Random forest is an ensemble machine learning method that uses multiple decision trees, while linear, simple linear, and polynomial regression are classical statistical methods.

---

## Question 2
What type of regression would be most appropriate for predicting carbon dioxide emissions when the independent variables considered are engine size and number of cylinders?

- **Multiple linear regression** ✓
- Logistic regression
- Simple linear regression
- Non-linear regression

**Answer:** Multiple linear regression

**Reason:** Multiple linear regression is used when there are two or more independent variables (engine size and number of cylinders) predicting a continuous target (CO2 emissions).

---

## Question 3
Why is ordinary least squares (OLS) regression's accuracy for complex data sets limited?

- OLS regression requires extensive tuning and hyperparameter adjustments.
- **OLS regression may inaccurately weigh outliers, resulting in skewed outputs.** ✓
- OLS regression cannot produce a best-fit line through the data.
- OLS regression is suited only for predicting categorical outcomes.

**Answer:** OLS regression may inaccurately weigh outliers, resulting in skewed outputs.

**Reason:** OLS minimizes squared errors, which gives outliers disproportionate weight since squaring amplifies large errors. This can significantly skew the results on complex datasets.

---

## Question 4
What multiple linear regression model estimates the values of coefficients by minimizing the Mean Squared Error (MSE)?

- Principal component analysis (PCA)
- **Ordinary least squares** ✓
- Gradient descent
- Stochastic gradient descent

**Answer:** Ordinary least squares

**Reason:** OLS is the fundamental method that mathematically calculates coefficients by minimizing the Mean Squared Error (MSE) between predicted and actual values.

---

## Question 5
What type of issue occurs when a high-degree polynomial regression model memorizes random noise in the data?

- Linear regression
- Gradient descent
- **Overfitting** ✓
- Underfitting

**Answer:** Overfitting

**Reason:** Overfitting occurs when a high-degree polynomial passes through every data point including noise, memorizing the training data instead of learning the underlying pattern. This leads to poor generalization on new data.

---

## Question 1
What is the primary purpose of logistic regression in machine learning?

- **Classify based on the predicted probability of an observation belonging to one of two classes** ✓
- Predict continuous values
- Create linear regression models
- Reduce the dimensionality of data

**Answer:** Classify based on the predicted probability of an observation belonging to one of two classes

**Reason:** Logistic regression is a binary classifier that predicts the probability of an observation belonging to one of two classes (e.g., yes/no, churn/stay) by using a threshold on the predicted probability.

---

## Question 2
What kind of outcomes does logistic regression predict?

- Only numerical values
- **Binary classification** ✓
- Multiple classes simultaneously
- Random outcomes without a clear pattern

**Answer:** Binary classification

**Reason:** Logistic regression predicts binary outcomes - it classifies observations into one of two classes (0 or 1, yes/no, true/false). It is specifically designed for binary classification problems.

---

## Question 3
Which parameter is used in logistic regression to determine the class of an observation?

- Linear regression
- Highest numerical value
- Mean of all observations
- **Threshold probability** ✓

**Answer:** Threshold probability

**Reason:** In logistic regression, a threshold probability (typically 0.5) is used to determine the class. If the predicted probability is greater than the threshold, the observation is assigned to class 1; otherwise, it is assigned to class 0.

---

## Question 4
What is the primary objective of the logistic regression training process?

- Randomly select parameters without any training
- Create multiple decision boundaries for classification
- **Minimize the cost function, or log-loss** ✓
- Achieve the highest possible accuracy in all classes

**Answer:** Minimize the cost function, or log-loss

**Reason:** The training process seeks to find the optimal parameters (θ) that minimize the cost function, which is log loss. Log loss measures how well the predicted probabilities match the actual classes.

---

## Question 5
A data scientist is using logistic regression to predict customer churn. After evaluating the model, they notice a high log-loss value. What is the most appropriate first step to improve the model?

- Use a different activation function
- **Parameter tuning** ✓
- More data
- Feature selection

**Answer:** Parameter tuning

**Reason:** High log-loss indicates the model's predictions are poor. The first step should be parameter tuning (adjusting learning rate, iterations, regularization) to better minimize the log-loss cost function before collecting more data or selecting features.

---

## Question 1
A company wants to forecast CO2 emissions based on engine size, number of cylinders, and fuel consumption. Which regression techniques should a company use?

- Simple regression
- Logistic regression
- Polynomial regression
- **Multiple regression** ✓

**Answer:** Multiple regression

**Reason:** Multiple regression is used when there are two or more independent variables (engine size, number of cylinders, fuel consumption) predicting a continuous target (CO2 emissions).

---

## Question 2
Which of the following examples best demonstrates when to use simple regression?

- Determining if customers belong to high, medium, or low-spending groups for product satisfaction.
- **Estimating plant growth based on the amount of daily sunlight reaching the Earth.** ✓
- Forecasting monthly sales based on historical sales data and economic indicators of the product.
- Analyzing the relationship between product price, reliability, quality, and customer satisfaction rating.

**Answer:** Estimating plant growth based on the amount of daily sunlight reaching the Earth.

**Reason:** Simple regression uses a single independent variable to estimate a dependent variable. Plant growth depends on sunlight (one variable), making it a simple regression case.

---

## Question 3
In which of the following cases would multiple linear regression be the most suitable analysis method?

- **Estimating sales based on budget, reach, and frequency.** ✓
- Predicting future sales based on last year's sales data.
- Analyzing customer sentiment using text-based feedback.
- Linking electricity consumption with changes in temperature.

**Answer:** Estimating sales based on budget, reach, and frequency.

**Reason:** Multiple linear regression uses multiple independent variables to predict an outcome. Sales depends on budget, reach, and frequency - three variables, making multiple regression appropriate.

---

## Question 4
Which scenario best demonstrates the appropriate use of logarithmic regression?

- **Estimating decrease in productivity gains with each additional hour of employee training.** ✓
- Analyzing the relationship between distance travel and fuel consumption.
- Modelling a proportional increase in sales with an increase in advertising budget.
- Predicting population growth where the growth rate increases over time.

**Answer:** Estimating decrease in productivity gains with each additional hour of employee training.

**Reason:** Logarithmic regression is used for diminishing returns - where incremental gains decrease as the input increases. This matches the scenario of decreasing productivity gains with more training hours.

---

## Question 5
A bank is using logistic regression to predict whether a loan applicant will default based on features such as age, loan amount, and repayment history. The model performs well but frequently classifies low-risk applicants as high-risk. What method could improve the model's accuracy in identifying the correct risk level?

- Perform cross-validation.
- **Adjust the decision threshold.** ✓
- Increase the model complexity.
- Add interaction terms to the model.

**Answer:** Adjust the decision threshold.

**Reason:** The default threshold is 0.5, but adjusting it (e.g., lowering it) can help reduce false positives and correctly identify low-risk applicants as low-risk. This directly addresses the misclassification issue.

---

## Question 6
A ride-sharing company uses logistic regression to predict whether a user will cancel a ride booking based on features like wait time and trip distance. For one booking, the model predicts a cancellation probability of 0.4. What does this probability indicate?

- The model predicts a 100% likelihood that the ride will be canceled.
- The model predicts a 60% likelihood that the ride will be canceled.
- **The model predicts a 40% likelihood that the ride will be canceled.** ✓
- There is no chance that the ride will be canceled.

**Answer:** The model predicts a 40% likelihood that the ride will be canceled.

**Reason:** A probability of 0.4 means there is a 40% chance the ride will be canceled. It does not mean certainty of cancellation or no chance of cancellation.

---

## Question 7
You are evaluating a binary classification model using log loss. Which of the following scenarios would result in the highest log loss?

- The model predicts a probability of 0.5 for both the correct and incorrect classes.
- **The model predicts a probability of 0.1 for the correct class and 0.9 for the incorrect class.** ✓
- The model predicts a probability of 0.7 for the correct class and 0.3 for the incorrect class.
- The model predicts a probability of 0.9 for the correct class and 0.1 for the incorrect class.

**Answer:** The model predicts a probability of 0.1 for the correct class and 0.9 for the incorrect class.

**Reason:** Log loss penalizes confident, incorrect predictions. When the model predicts 0.9 for the incorrect class (very confident but wrong), the log loss is highest because it heavily penalizes this confident错误 prediction.
