# Module 5 Quiz Answers

---

## Question 1
**What is the purpose of a train/test split in machine learning?**

- To estimate the performance of machine learning algorithms on unseen data
- To optimize the model's hyperparameters
- To increase the size of the training dataset
- To visualize the dataset

**Answer**: To estimate the performance of machine learning algorithms on unseen data

---

## Question 2
**What does the F1 score represent in model evaluation?**

- The harmonic mean of precision and recall
- The weighted sum of precision and recall
- The probability of a true positive prediction
- The average accuracy of the model

**Answer**: The harmonic mean of precision and recall

---

## Question 3
**What does R-squared measure in regression analysis?**

- The total variance of the target variable
- The unexplained variance of the model
- The proportion of variance in the target variable explained by the model
- The sum of squared differences between predicted and actual values

**Answer**: The proportion of variance in the target variable explained by the model

---

## Question 4
**What is the effect of using a mean-value model in regression analysis?**

- Explained variance will be greater than total variance
- R-squared will be one
- R-squared will be one half
- R-squared will be zero, as the model explains no variance

**Answer**: R-squared will be zero, as the model explains no variance

---

## Question 5
**Which of the following clustering evaluation metrics ranges from -1 to 1, with higher values indicating better-defined clusters?**

- Silhouette Score
- Davies-Bouldin Index
- Within-cluster sum of squares (WCSS)
- Inertia

**Answer**: Silhouette Score

---

## Question 6
**Which of the following best describes the role of the test set in model validation?**

- The test set is used to transform the target variable.
- **The test set is used to evaluate the model after it has been trained and validated.** ✓
- The test set is used to optimize hyperparameters.
- The test set is used to train the model.

**Answer**: The test set is used to evaluate the model after it has been trained and validated.

---

## Question 7
**What is the benefit of transforming a skewed target variable, such as using a Box-Cox or logarithmic transformation?**

- It increases the complexity of the model.
- It guarantees better model performance on training data.
- **It makes the model fit the target variable more easily by reducing skewness.** ✓
- It improves the interpretability of the model's predictions.

**Answer**: It makes the model fit the target variable more easily by reducing skewness.

---

## Question 8
**In the context of regularization, what does the lambda parameter control?**

- The number of iterations in the training process
- **The penalty term's influence on the cost function** ✓
- The learning rate of the model
- The number of features in the data set

**Answer**: The penalty term's influence on the cost function

---

## Question 9
**In which scenario is Lasso regression most beneficial?**

- When the model requires high computational efficiency
- When the data set is very small
- When all features are equally important
- **When the data is sparse and has many irrelevant features** ✓

**Answer**: When the data is sparse and has many irrelevant features

---

## Question 10
**What is the primary cause of data leakage in machine learning?**

- Using a larger data set for training
- Not scaling the features correctly
- Using outdated information
- **Using features that would not be available to a deployed model in the real world** ✓

**Answer**: Using features that would not be available to a deployed model in the real world

---

## Question 11
**In supervised learning evaluation, which metric would be the most useful when the cost of false positives is high?**

- Recall
- Accuracy
- **Precision** ✓
- F1 Score

**Answer**: Precision

---

## Question 12
**Which regression evaluation metric measures the average absolute difference between the predicted values and the observed values?**

- **Mean Absolute Error (MAE)** ✓
- Root Mean Squared Error (RMSE)
- Mean Squared Error (MSE)
- R-squared

**Answer**: Mean Absolute Error (MAE)

---

## Question 13
**A company wants to measure the consistency of its clustering algorithm by examining how well-separated the individual clusters are. Which of the following metrics is most suitable for this purpose?**

- Evaluation method
- Davies-Bouldin index
- **Silhouette score** ✓
- Elbow method

**Answer**: Silhouette score

---

## Question 14
**A machine learning engineer selected hyperparameters based on performance on the test set. What risk does this introduce?**

- Cross-validation
- **Data snooping** ✓
- Overfitting
- Train-test split

**Answer**: Data snooping

---

## Question 15
**During a model performance review, a machine learning engineer compares two regularization techniques and observes that one shrinks coefficients to zero while the other reduces their size. What is the key difference between Lasso and Ridge regression?**

- Lasso can only be used for feature selection, while Ridge is used for general regression tasks.
- Ridge uses an L-1 penalty, and Lasso uses an L-2 penalty.
- Lasso uses a larger dataset than Ridge.
- **Lasso uses an L-1 penalty, and Ridge uses an L-2 penalty.** ✓

**Answer**: Lasso uses an L-1 penalty, and Ridge uses an L-2 penalty.

---

## Question 16
**Cristin, a machine learning engineer, is training a time-sensitive model and mistakenly included future timestamps in the training set. Which of the following would prevent data leakage?**

- **Excluding features that rely on future data during training** ✓
- Using global averages as features in the training data
- Reusing the dataset for both training and testing the model
- Ensuring that future data is incorporated into the training data

**Answer**: Excluding features that rely on future data during training

---

## Question 17
**An analyst uses a model that ranks feature importance, ignoring how some features work together to influence predictions. What pitfall might occur?**

- **Failing to account for interactions between features** ✓
- Using only one model to calculate feature importance
- Assuming feature importance indicates causality
- Scaling features before assessing importance

**Answer**: Failing to account for interactions between features
