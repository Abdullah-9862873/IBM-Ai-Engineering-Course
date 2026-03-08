# Module 6 Quiz Answers - Final Project

---

## Question 1
**What is the True Positive rate of the RandomForestClassifier based on the confusion matrix from Exercise 13?**

- The True Positive rate (TPR) for the RandomForestClassifier is approximately **50%** (or rounded to nearest whole number: **50%**)

---

## Question 2
**Identify the most important feature for predicting whether it will rain based on the feature importance bar graph.**

- **Humidity3pm** is the most important feature for predicting whether it will rain

---

## Question 3
**Which of the following features would be inefficient in predicting whether it will rain tomorrow or not?** (Select all that apply)

- **Evaporation** - This feature relies on the entire duration of today's data to calculate, making it inefficient for predicting tomorrow's rain
- **WindGustDir** - Direction of strongest gust also depends on full day measurements

Note: Humidity9am, MaxTemp can be useful predictors as they are available at the start of the day.

---

## Question 4
**In bullets or as a numbered list, answer the following:**
- Comment on the accuracy of the Logistic Regression and Random Forest Classifier models
- Comment on the true positive rate of the two models
- Overall, which one of the two is a better predictor of whether it will rain tomorrow or not?

### Answers:

- **Accuracy Comparison:**
  - Random Forest Classifier typically achieves higher accuracy (~84-85%) compared to Logistic Regression (~82-83%)
  - Random Forest handles non-linear relationships and feature interactions better

- **True Positive Rate Comparison:**
  - Random Forest Classifier: ~50% TPR (better at identifying rainy days)
  - Logistic Regression: ~45-48% TPR (lower true positive rate)
  - Both models have similar precision, but Random Forest has slightly better recall/TPR

- **Better Predictor:**
  - **Random Forest Classifier is the better predictor** for whether it will rain tomorrow
  - Reasons:
    - Higher overall accuracy
    - Better true positive rate (better at catching actual rainy days)
    - Handles complex, non-linear relationships between weather features
    - Better at capturing feature interactions that are important for rainfall prediction

---

## Additional Notes from Final Project

### Points to Note - 1 (Data Leakage)
Features that would be inefficient for predicting tomorrow's rainfall:
- **Evaporation** - Requires full day measurements
- **Sunshine** - Depends on complete day of data
- Features that rely on entire duration of today for their evaluation

### Points to Note - 2 (True Positive Rate)
The true positive rate (recall for "Yes" class) can be calculated from the confusion matrix:
- TPR = True Positives / (True Positives + False Negatives)

### Points to Note - 3 (Most Important Feature)
The most important feature is typically **Humidity3pm** - afternoon humidity is highly correlated with rainfall prediction

### Points to Note - 4 (Model Comparison)
- Random Forest typically has higher accuracy (~84%) and better TPR (~50%)
- Logistic Regression has slightly lower accuracy (~82%) and lower TPR (~45-48%)
- Random Forest is the better choice for this rainfall prediction task

---

## Additional Quiz Questions

---

## Question 1
**A finance company has designed a model differentiating fraud, non-fraud, and suspicious categories for real-time transaction alerts. How does the one-versus-all classification strategy help them to construct a multi-class model using binary classifiers?**

- **To extend binary classifiers to handle multiple classes** ✓
- To improve the performance of neural networks
- To handle binary classification only
- To classify data into its most important classes

**Answer**: To extend binary classifiers to handle multiple classes

---

## Question 2
**You are building a regression tree to predict house prices in a skewed market where a few houses are expensive. Why should you use the median instead of the mean at leaf nodes to improve prediction accuracy?**

- The mean is always inaccurate.
- **The median is less affected by skewed data.** ✓
- The median results in lower mean squared errors (MSE).
- The mean is more expensive to compute.

**Answer**: The median is less affected by skewed data

---

## Question 3
**Imagine you are predicting house prices in a city using a decision tree model. You notice that the model starts to fit the training data more closely. How do bias and variance change as you increase the complexity of your decision tree?**

- **Bias decreases, and variance increases** ✓
- Both bias and variance remain constant
- Bias increases, and variance decreases
- Both bias and variance decrease

**Answer**: Bias decreases, and variance increases

---

## Question 4
**You are managing a network security system and want to enhance monitoring by identifying activity that doesn't match usual patterns. Machine learning can assist with this goal. Which machine learning task is the most relevant?**

- **Spotting traffic patterns that deviate from the norm** ✓
- Predicting future network usage based on historical trends
- Categorizing traffic into streaming, browsing, or gaming types
- Grouping users based on their browsing habits for analysis

**Answer**: Spotting traffic patterns that deviate from the norm

---

## Question 5
**A climate scientist is plotting a curve-shaped rise in global temperatures based on CO₂ concentration over time. Which regression method best captures the non-linear but smooth curvature in the data?**

- Linear regression
- **Polynomial regression** ✓
- Logarithmic regression
- Exponential regression

**Answer**: Polynomial regression

---

## Question 6
**Which of the following supervised learning models would be most appropriate for a binary classification task that aims to classify customers as either likely to purchase a product or not, based on their proximity to similar customers?**

- **K-nearest neighbors (KNN)** ✓
- Decision tree
- Support vector machine (SVM)
- Logistic regression

**Answer**: K-nearest neighbors (KNN)

---

## Question 7
**What characteristic makes PCA suitable as a preprocessing step for clustering high-dimensional datasets?**

- Identifies nonlinear dependencies to create clusters
- **Converts correlated input features into orthogonal components** ✓
- Removes outliers and rescales values for each feature
- Assigns cluster labels to each observation automatically

**Answer**: Converts correlated input features into orthogonal components

---

## Question 8
**A data scientist proposed that training a logistic regression model on a full dataset will take longer than expected. They decided to use a method that approximate gradients using smaller samples to accelerate training. What is this method?**

- **Stochastic gradient descent** ✓
- Grid search
- Least squares regression
- Backpropagation

**Answer**: Stochastic gradient descent

---

## Question 9
**The healthcare industry wants to predict which patients need urgent treatment using a logistic regression model. However, many low-risk patients are flagged as high-risk, leading to low manpower. How can the healthcare industry reduce the false positives in the model?**

- Train the model on fewer features
- Normalize patient data
- Use the linear regression method
- **Fine-tune the decision boundary** ✓

**Answer**: Fine-tune the decision boundary

---

## Question 10
**A data scientist analyzes customer transaction data and wants to identify natural groupings based on transaction frequency and purchase amounts. They are interested in a method that starts with each customer as an individual cluster and then merges similar customers into larger clusters. Which clustering method should they use?**

- Partition-based clustering
- Density-based clustering
- **Agglomerative clustering** ✓
- Divisive clustering

**Answer**: Agglomerative clustering

---

## Question 11
**Which of the following scenarios is least appropriate for DBSCAN due to its focus on spatial clustering?**

- Analyzing vegetation density in satellite images
- Grouping daily travel behaviors in a city
- **Forecasting future customer spending trends** ✓
- Identifying outliers in machine sensor data

**Answer**: Forecasting future customer spending trends

---

## Question 12
**Which dimensionality reduction technique is most suitable for clustering applications when preserving only the local data structure is important?**

- Linear Discriminant Analysis (LDA)
- **t-distributed Stochastic Neighbor Embedding (t-SNE)** ✓
- Principal Component Analysis (PCA)
- Uniform Manifold Approximation and Projection (UMAP)

**Answer**: t-distributed Stochastic Neighbor Embedding (t-SNE)

---

## Question 13
**Which tool should you leverage for building a neural network with a deep learning framework that supports GPU acceleration and flexible experimentation?**

- Pandas
- Scikit-learn
- Matplotlib
- **PyTorch** ✓

**Answer**: PyTorch

---

## Question 14
**A machine learning engineer is building a model to detect spam emails. Why would a machine learning model be more suitable than a rule-based filter?**

- It only uses labeled spam emails.
- It avoids feature selection.
- It doesn't require a dataset.
- **It learns email patterns automatically.** ✓

**Answer**: It learns email patterns automatically

---

## Question 15
**Taniya is designing a machine learning pipeline and wants a tool to preprocess data, build models, validate performance, and export results for deployment within a Python ecosystem. Which tool best supports this end-to-end functionality?**

- NumPy
- Matplotlib
- **Scikit-learn** ✓
- Pandas

**Answer**: Scikit-learn
