# Quiz Answers

## Question 1
What type of machine learning method is classification?

- **Supervised** ✓
- Reinforcement
- Semi-supervised
- Unsupervised

**Answer:** Supervised

**Reason:** Classification is a supervised learning method because it uses labeled data (with known output classes) to train a model that can predict labels for new, unlabeled data.

---

## Question 2
What is the purpose of using a one-versus-all classification strategy?

- **To extend binary classifiers to handle multiple classes** ✓
- To handle binary classification
- To improve the performance of neural networks
- To classify data into one or more classes

**Answer:** To extend binary classifiers to handle multiple classes

**Reason:** One-vs-All (One-vs-Rest) creates k binary classifiers for k classes, where each classifier distinguishes one class from all others, enabling binary classifiers to handle multi-class problems.

---

## Question 3
In a decision tree, what does each leaf node represent?

- **A class label** ✓
- The split criterion used at that level
- The result of a test
- A feature of the data

**Answer:** A class label

**Reason:** In a decision tree, leaf nodes (terminal nodes) represent the final classification outcome - the class label assigned to data points that reach that leaf.

---

## Question 4
Why might you want to prune a decision tree?

- To maximize the number of features used in the tree
- To ensure the tree has no leaf nodes
- To increase the complexity of the model
- **To avoid overfitting the training data** ✓

**Answer:** To avoid overfitting the training data

**Reason:** Pruning removes branches that don't significantly improve performance, simplifying the tree and making it more generalizable to new data. This prevents overfitting.

---

## Question 5
What is the primary goal of a regression tree?

- To minimize the number of leaf nodes
- To classify data into discrete categories
- **To predict continuous values based on features** ✓
- To split data based on information gain and entropy

**Answer:** To predict continuous values based on features

**Reason:** A regression tree predicts continuous values (like temperature, salary, revenue) rather than discrete class labels. It uses the average of target values in leaf nodes for predictions.

---

## Question 1
What does the k parameter in k-NN represent?

- **The number of nearest neighbors used for prediction** ✓
- The distance measure used to calculate similarity
- The number of classes in the data
- The number of features in the data

**Answer:** The number of nearest neighbors used for prediction

**Reason:** In k-NN, k represents the number of nearest neighbors that are considered when making a prediction. For classification, it uses majority vote from these k neighbors; for regression, it uses the average/median of their values.

---

## Question 2
What is the primary purpose of scaling features before applying k-NN?

- **To ensure all features contribute equally to the distance measure** ✓
- To reduce the total number of features
- To make computation faster
- To give more weight to features with higher values

**Answer:** To ensure all features contribute equally to the distance measure

**Reason:** Features with large values would dominate the distance calculation in k-NN, causing biased predictions. Scaling features (standardization) removes this artificial importance and ensures all features contribute equally.

---

## Question 3
What is the primary goal of an SVM classifier in a binary classification task?

- To minimize the number of support vectors needed
- To find multiple hyperplanes for each class
- To find a line that passes through the majority of data points
- **To create a hyperplane that maximizes the margin between two classes** ✓

**Answer:** To create a hyperplane that maximizes the margin between two classes

**Reason:** SVM finds a hyperplane that distinctly separates two classes while maximizing the margin (distance to nearest points). The larger the margin, the better the model's accuracy on new, unseen data.

---

## Question 4
What is the role of the C parameter in SVM?

- It defines the kernel function used for the SVM
- It determines the dimensionality of the data space
- **It controls the width of the margin by allowing some misclassifications** ✓
- It sets the number of support vectors

**Answer:** It controls the width of the margin by allowing some misclassifications

**Reason:** The C parameter in SVM controls the tradeoff between maximizing the margin and minimizing misclassifications. Small C = softer margin (more misclassifications allowed); Large C = harder margin (stricter separation).

---

## Question 5
What does bias refer to in the context of predictive modeling?

- **The average difference between predicted values and actual target values** ✓
- The degree of complexity of the model
- The number of support vectors in a model
- The variability of the model's predictions across different datasets

**Answer:** The average difference between predicted values and actual target values

**Reason:** Bias measures how far off (on average) a model's predictions are from the actual values. High bias means the model is inaccurate (underfitting), while zero bias means perfect predictions.

---

## Question 1
The healthcare industry uses patient history to classify diseases into multiple categories. Which classification model should they use in this task?

- **Decision tree** ✓
- Naïve Bayes
- K-nearest neighbors
- Logistic Regression

**Answer:** Decision tree

**Reason:** Decision trees are excellent for classification tasks with multiple categories. They can handle multiple features and create clear decision paths for classifying diseases into different categories based on patient history.

---

## Question 2
Which voting scheme can be used to assign the final label in a one-versus-one classification approach?

- **Popularity vote** ✓
- Probability weighing
- One-hot encoding
- Random selection

**Answer:** Popularity vote

**Reason:** In one-vs-one classification, a voting scheme is used where each binary classifier votes for a class, and the class with the most votes (popularity vote) is selected as the final label.

---

## Question 3
Which feature split would likely provide the highest information gain in a decision tree?

- **A feature that decreases entropy in the resulting nodes** ✓
- A feature that is categorical only
- A feature that maximizes entropy within each node
- A feature that creates the most balanced branches

**Answer:** A feature that decreases entropy in the resulting nodes

**Reason:** Information gain measures the decrease in entropy after a split. The higher the information gain, the better the split at reducing uncertainty. A feature that decreases entropy in resulting nodes provides the highest information gain.

---

## Question 4
A data scientist splits a continuous feature using thresholds between consecutive sorted values. Which method is being used to determine candidate splits in this regression tree?

- Mean squared error (MSE) method
- **Midpoints method** ✓
- Exhaustive search method
- Entropy reduction method

**Answer:** Midpoints method

**Reason:** The midpoints method defines candidate thresholds as the midpoints between consecutive sorted values of the feature. This is a common strategy for determining candidate splits in regression trees.

---

## Question 5
Why do you observe poor accuracy in K-nearest neighbors (KNN) predictions after increasing the K value?

- Too small training data
- **Too many smoothing of patterns** ✓
- Too many scaling errors
- Too many irrelevant features

**Answer:** Too many smoothing of patterns

**Reason:** When K is too large, KNN averages over too many neighbors, smoothing out finer details and local patterns. This causes underfitting, leading to poor accuracy on both training and test data.

---

## Question 6
Rommy is using support vector regression (SVR) to predict housing prices. He adjusts the epsilon (ε) value and notices that more data points are falling outside the shaded prediction band. What does this change in epsilon represent?

- It modifies the kernel function used for transformation.
- **It changes the width of the margin around the predicted curve.** ✓
- It affects how features are scaled in input space.
- It increases the number of support vectors used.

**Answer:** It changes the width of the margin around the predicted curve.

**Reason:** In SVR, epsilon (ε) defines the tube around the prediction curve. A larger epsilon creates a wider tube, meaning more points are considered within the margin (not errors), while a smaller epsilon makes the tube tighter.

---

## Question 7
Your team is evaluating the model and has identified that it makes inaccurate predictions even on the training data and seems too simple. What is the consequence of high bias in a model?

- **The model performs poorly on training data due to oversimplification.** ✓
- The model performs well on both training and testing datasets.
- The model tends to overfit the training data for predictions.
- The model is sensitive to noise in the training data.

**Answer:** The model performs poorly on training data due to oversimplification.

**Reason:** High bias means the model is too simple (underfitting) and cannot capture the underlying patterns in the data. This results in poor performance on both training and test data.
