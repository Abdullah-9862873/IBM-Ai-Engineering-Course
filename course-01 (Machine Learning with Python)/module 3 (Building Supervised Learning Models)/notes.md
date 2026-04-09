# Introduction to Classification

## What is Classification?
-> Supervised machine learning method
-> Uses fully trained models to predict labels on new data
-> Labels form a **categorical variable** with discrete values
-> Model adjusts data to fit algorithm and classifies accordingly

### Supervised Learning:
-> Aims to understand data in correct context when answering a specific question
-> Ensures data accuracy when making predictions
-> Input and predicted output are defined

---

## Applications of Classification

### Common Use Cases:
| Application | Description |
|-------------|-------------|
| **Email Filtering** | Classify emails as spam or not spam |
| **Speech-to-Text** | Convert speech to text |
| **Handwriting Recognition** | Recognize handwritten characters |
| **Biometric Identification** | Identify individuals based on physical features |
| **Document Classification** | Categorize documents |
| **Churn Prediction** | Predict if customer will discontinue service |
| **Customer Segmentation** | Predict category customer belongs to |
| **Advertising Response** | Predict if customer will respond to campaign |

---

## Binary Classification

### What is Binary Classification?
-> Predicts between **two possible classes**
-> Example: Loan default prediction

### Example: Bank Loan Default
-> Use historical loan default data
-> Features: age, income, credit debt levels
-> Target: Will customer default (Yes/No)
-> Given new customer data, predict likelihood of default

---

## Multi-Class Classification

### What is Multi-Class Classification?
-> Predicts among **more than two classes**
-> Example: Drug prescription

### Example: Drug Prescription
-> Patients suffered from same illness
-> Each patient responded positively to one of three medications
-> Use labeled dataset to build classification model
-> Predict which drug might work for future patient

---

## Classification Algorithms

### Common Classification Algorithms:
- Naive Bayes
- Logistic Regression
- Decision Trees
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Neural Networks

### Algorithm Capabilities:
-> Algorithms like Logistic Regression, KNN, and Decision Trees can distinguish **multiple classes**
-> Many classification algorithms can only handle **two classes** (binary)
-> Use strategies to extend binary classifiers to multi-class

---

## Multi-Class Classification Strategies

### 1. One-vs-All (One-vs-Rest)

#### How It Works:
-> Implements **k binary classifiers** (where k = number of classes)
-> Each classifier is assigned a single label (target class)
-> Each classifier predicts: "Does it have this label?" (Yes/No)

#### Process:
1. Train classifier for each class
2. Each classifier distinguishes one class from all others
3. Given new data point, all classifiers make predictions
4. Select class with **highest confidence/probability**

#### Properties:
-> If k classes, exactly k binary classifiers
-> Some data points might not belong to any class (not picked by any classifier)
-> Useful for identifying outliers or noise
-> Unclassified points fall into another category

---

### 2. One-vs-One

#### How It Works:
-> Consider **all possible pairs of classes**
-> For each pair, train a classifier on subset of data with those two labels
-> Each classifier decides: "Is it this or is that?"

#### Process:
1. For k classes, create k(k-1)/2 classifiers
2. Each classifier trained on just two classes
3. Final class decided by **voting scheme**

#### Voting Schemes:
| Scheme | Description |
|--------|-------------|
| **Simple Popularity** | Class predicted by most classifiers wins |
| **Confidence Weighted** | Vote weighed by confidence/probability assigned |

#### Handling Ties:
-> If equal votes, use confidence-weighted scheme
-> Or try one-vs-all classification instead

---

## Summary:
-> Classification is supervised ML method to predict categorical labels
-> Used for churn prediction, customer segmentation, loan default, drug prescription
-> Common algorithms: Naive Bayes, Logistic Regression, Decision Trees, KNN, SVM, Neural Networks
-> Binary classification: 2 classes; Multi-class: more than 2 classes
-> One-vs-All: k classifiers, one per class
-> One-vs-One: k(k-1)/2 classifiers, pairwise comparison

---

# Decision Trees for Machine Learning

## What is a Decision Tree?
-> Algorithm that can be visualized as a flowchart
-> Used for classifying data points
-> Each internal node corresponds to a **test**
-> Each branch corresponds to the **result of the test**
-> Each terminal (leaf) node **assigns data to a class**

### Visualization:
```
       [Test Feature]
        /    |    \
   Result1 Result2 Result3
      |      |       |
   [Node]  [Node]  [Leaf] -> Class
```

---

## Example: Drug Prediction

### Dataset Features:
-> Age (young, middle-aged, senior)
-> Gender (male, female)
-> Blood pressure
-> Cholesterol (high, normal)

### Target:
-> Drug A or Drug B

### Decision Logic:
```
Age = middle-aged → Drug B
Age = young AND gender = male → Drug B
Age = senior AND cholesterol = normal → Drug B
Age = young AND gender = female → Drug A
Age = senior AND cholesterol = high → Drug A
```

---

## How to Build Decision Trees

### Training Process:
1. **Start with seed node** and labeled training data
2. **Train node** on assigned data by finding best feature to split
3. **Split data** according to a pre-selected splitting criterion
4. **Pass partitions** along branches to new nodes
5. **Repeat** process for each new node
6. **Grow until**:
   - All nodes contain single class, OR
   - Run out of features, OR
   - Pre-selected stopping criterion met

### Key Concept:
-> Each feature can only be used **once** in the path from root to leaf

---

## Stopping Criteria (Pre-pruning)

### Common Stopping Criteria:
| Criterion | Description |
|-----------|-------------|
| **Max Tree Depth** | Maximum depth reached |
| **Min Samples in Node** | Minimum data points in a node exceeded |
| **Min Samples in Leaf** | Minimum samples in leaf node exceeded |
| **Max Leaf Nodes** | Maximum number of leaf nodes reached |

---

## Tree Pruning

### Why Prune?
-> If tree is too complex, may be **overfitting** to training data
-> Too many classes and features may capture **noise and irrelevant details**
-> Pruning simplifies the model and improves **generalization**

### Benefits of Pruning:
-> More concise and easier to understand
-> Better **predictive accuracy**
-> Avoids overfitting

### Types:
-> **Pre-pruning**: Stop tree from growing early (using stopping criteria)
-> **Post-pruning**: Cut branches that don't significantly improve performance

---

## Splitting Criterion

### Purpose:
-> Select feature that **best splits the data** at each node
-> Measure split quality for determining best split

### Two Common Measures:
1. **Information Gain** (Entropy reduction)
2. **Gini Impurity**

---

## Entropy

### What is Entropy?
-> Measure of **information disorder** or randomness in a dataset
-> Measures how **random** the classes in a node are
-> Measures **uncertainty** of feature split result

### Goal:
-> Find trees with **smallest entropy** in nodes
-> As entropy decreases, information gain increases

### Formula:
```
Entropy = -pA × log2(pA) - pB × log2(pB)

Where:
- pA = proportion of class A in node
- pB = proportion of class B in node
```

### Entropy Values:
| Scenario | Entropy Value |
|----------|---------------|
| Classes completely homogenous | 0 |
| Classes equally divided | 1 |

### Example:
-> If pA = 0.5 and pB = 0.5 → Entropy = 1 (maximum randomness)

---

## Information Gain

### What is Information Gain?
-> **Entropy of tree before split** minus **weighted entropy after split**
-> Measures how much **certainty** is gained by a split

### Formula:
```
Information Gain = Entropy(before split) - Weighted Entropy(after split)
```

### Key Property:
-> **Constructing a Decision Tree = finding features with highest information gain**
-> As entropy decreases, information gain increases
-> Best split = feature with **highest information gain**

### Example:
-> Using cholesterol to split all patients yields information gain of 0.042

---

## Gini Impurity

### What is Gini Impurity?
-> Another measure for split quality
-> Measures probability of **incorrect classification**
-> Range: 0 to 0.5 (for binary classification)

### Goal:
-> Find features that **decrease impurity** in leaves

---

## Decision Tree Example: Drug Prescription

### Dataset:
-> 14 patients with illness
-> Features: Sex, Cholesterol, Age
-> Target: Drug A or Drug B

### Building Process:
1. Start with all patients at root node
2. **First split**: Try cholesterol → not best (impurity still high)
3. **Second split**: Try sex → better separation
   - Females → mostly Drug B
   - Males → need further splitting
4. **Third split** (males): Use cholesterol → pure terminal leaves

### Result:
-> Terminal leaves contain patients of single class
-> Tree stops when stopping criterion met

---

## Advantages of Decision Trees

### 1. Visualizable
-> Can see exactly how decisions are made
-> Highly **interpretable**

### 2. Feature Importance
-> Can gain insights about how **predictive** each feature is
-> Tree grows by selecting next best feature to split

### 3. Easy to Understand
-> Simple flowchart-like structure
-> Non-technical stakeholders can understand

### 4. Handles Both Numerical and Categorical Data
-> No need for extensive data preprocessing

---

## Summary:
-> Decision Tree = flowchart-like algorithm for classifying data points
-> Internal nodes = tests, branches = test results, leaf nodes = classes
-> Build by splitting data using best feature at each node
-> Stop growing when criteria met (pre-pruning)
-> Prune to avoid overfitting and improve generalization
-> Split measures: Information Gain (entropy) and Gini Impurity
-> Higher information gain = better split = more certain
-> Advantages: Visualizable, interpretable, feature importance insights

---

# Regression Trees

## What is a Regression Tree?
-> Analogy to decision tree but predicts **continuous values** instead of discrete classes
-> Adapted from decision tree to solve **regression problems**

### Key Difference: Classification vs Regression

| Aspect | Classification Tree | Regression Tree |
|--------|---------------------|-----------------|
| **Target Variable** | Categorical (discrete) | Continuous |
| **Prediction** | Class-labeled majority vote | Average of target values |
| **Example** | Spam detection, medical diagnosis | Revenue, temperature, wildfire risk |

---

## Use Cases

### Classification Trees:
-> Spam detection
-> Image classification
-> Medical diagnosis

### Regression Trees:
-> Predicting revenue
-> Predicting temperatures
-> Predicting wildfire risk

---

## How Regression Trees Work

### Basic Concept:
-> Recursively split dataset into subsets
-> Maximize information gained from data splitting
-> Minimize randomness in split nodes

### Splitting Process:
1. Given a continuous feature and threshold value (α)
2. Split data into two subsets:
   - Data > α → Right node
   - Data < α → Left node
3. For binary features: split according to the two classes
4. Make prediction at each node

---

## Prediction in Regression Trees

### How to Predict:
-> Predicted value (ŷ) for a node = **average of actual target values** in that node

### Formula:
```
ŷ = (1/n) × Σyi

Where:
- n = number of data points in node
- yi = actual target values
```

### Alternative:
-> Can also use **median** value (preferred for skewed data)
-> For normally distributed data, median is comparable to mean

---

## Split Quality Measure

### Problem with Classification Criteria:
-> Cannot use entropy or information gain (for classification)
-> Need different measure for continuous targets

### Solution: Mean Squared Error (MSE)

### Why MSE?
-> Measures **variance** of target values within each node
-> Smaller variance = values agree more closely
-> Goal: Minimize variance in predicted values

### Formula:
```
MSE = (1/n) × Σ(yi - ŷ)²
```

---

## Weighted Average of MSE

### Purpose:
-> Measure overall quality of a split

### Formula:
```
Weighted MSE = (nL × MSEL + nR × MSER) / (nL + nR)

Where:
- nL = number of observations in left split
- nR = number of observations in right split
- MSEL = MSE of left split
- MSER = MSE of right split
```

### Key Point:
-> **Lower weighted MSE = lower variance = higher split quality**

---

## Training Process

### Step-by-Step:
1. For each feature, try different threshold values
2. Calculate MSE for left and right subsets
3. Calculate weighted MSE for the split
4. Select split with **lowest weighted MSE**
5. Repeat for each node

### Goal:
-> Minimize variance in predicted values
-> Improve accuracy of regression tree

---

## Binary Features

### Process:
-> Data separated into its two classes
-> Split quality = weighted average of class MSEs
-> Only one possible result → already optimized

---

## Multi-Class Features

### Strategy:
-> Use One-vs-One or One-vs-All to generate binary splits
-> For each binary split, calculate weighted MSE
-> Select split that **minimizes weighted MSE**

### Result:
-> Lowest prediction variance
-> Best split selected

---

## Choosing Trial Thresholds

### Strategy:
1. **Sort** feature values (Xi ≤ Xj for i < j)
2. **Remove duplicates** (Xi < Xj for all i < j)
3. **Define candidate thresholds** as midpoints:
   ```
   αi = (Xi + Xi+1) / 2
   ```
4. **Choose threshold** that minimizes weighted MSE

### Considerations:
| Scenario | Approach |
|----------|----------|
| **Small data** | Exhaustive search (all thresholds) |
| **Large data** | Sample subset of thresholds (efficient but less accurate) |
| **Non-uniform distribution** | Consider distribution when sampling |

### Limitation:
-> Exhaustive search doesn't scale well to big data

---

## Summary:
-> Regression Tree = Decision tree for continuous variables
-> Target: Classification = categorical, Regression = continuous
-> Prediction at leaf = average of target values in node
-> Split quality measured using MSE (variance)
-> Select feature/threshold with lowest weighted MSE
-> Binary features: already optimized
-> Multi-class: use One-vs-One or One-vs-All strategies
-> Choose thresholds based on data size and distribution

---

# Supervised Learning with SVMs

## What is SVM (Support Vector Machines)?
-> Supervised learning technique for building classification and regression models
-> Maps each data instance as a point in **multidimensional space**
-> Input features are coordinates in that space
-> Classifies data by identifying a **hyperplane** that differentiates two classes

---

## How SVM Works

### Basic Concept:
-> SVM classifies input data by identifying a **hyperplane**
-> Hyperplane distinctly separates two classes
-> Points on either side of hyperplane belong to different classes
-> **Largest margin** = better model accuracy on new data

### Example:
```
        Class 1 (above line)
    ○ ○ ○ ○ ○ ○ ○
    -----------------  ← Hyperplane (decision boundary)
    ■ ■ ■ ■ ■ ■ ■
        Class 0 (below line)
```

---

## Key Terminology

### 1. Hyperplane
-> Decision boundary that separates classes
-> In 2D: a **line**
-> In 3D: a **plane**
-> In n-dimensions: a **hyperplane**

### 2. Margin
-> Distance from hyperplane to **closest points** from each class
-> **Larger margin** = better generalization

### 3. Support Vectors
-> Nearest-point representatives from each class
-> These points define the margin
-> Critical for creating the hyperplane

---

## Soft Margin vs Hard Margin

### Hard Margin:
-> Requires perfect separation of classes
-> Only works when data is **linearly separable**
-> Sensitive to noise

### Soft Margin:
-> Allows **misclassifications**
-> Tolerates some errors to maximize margin
-> More robust to noisy data

### Parameter C (Regularization):
| C Value | Effect |
|---------|--------|
| **Small C** | Softer margin, allows more misclassifications |
| **Large C** | Stricter margin, fewer misclassifications |

---

## SVM for Classification

### Process:
1. Plot data points in feature space
2. Find hyperplane that maximizes margin
3. New data classified based on which side of hyperplane it falls

### Making Predictions:
```
If (w · x + b) > 0 → Class 1
If (w · x + b) < 0 → Class 0

Where:
- w = weight vector
- x = input features
- b = bias term
```

---

## Nonlinear SVM (Kernel Trick)

### Problem:
-> Some classes are not linearly separable (e.g., concentric circles)

### Solution:
-> Map data into **higher-dimensional space**
-> Find a hyperplane in higher dimension to separate classes

### Example:
-> 2D data with circular classes
-> Transform to 3D using polynomial feature
-> Classes become separable by a plane

### This is called **Kerneling**

---

## Kernel Functions

### Available Kernels in Scikit-learn:

| Kernel | Description |
|--------|-------------|
| **Linear** | Default, corresponds to usual SVM model |
| **Polynomial (Poly)** | Parabolic embedding |
| **RBF (Radial Basis Function)** | Scores high for close points, decreases with distance |
| **Sigmoid** | Same function as logistic regression |

### Choosing Kernel:
-> No straightforward way to know which kernel performs best
-> Try different kernels and compare results

---

## SVM for Regression (SVR)

### Concept:
-> Support Vector Regression (SVR)
-> Uses kernel functions (e.g., RBF)

### Epsilon Tube:
-> Shaded region around prediction curve
-> Points inside tube = signal (not errors)
-> Points outside tube = noise

### Epsilon Parameter:
| Epsilon Value | Effect |
|---------------|--------|
| **Smaller** | Tighter tube, more points considered as noise |
| **Larger** | Looser tube, more tolerance for errors |

---

## SVM Applications

### When to Use SVM:
| Application | Description |
|-------------|-------------|
| **Image Classification** | Image recognition, handwritten digit recognition |
| **Spam Detection** | Classify emails as spam or not |
| **Sentiment Analysis** | Parse text for opinions/emotions |
| **Speech Recognition** | Convert speech to text |
| **Anomaly Detection** | Identify outliers |
| **Noise Filtering** | Remove noise from data |

---

## Advantages of SVM

### 1. Effective in High-Dimensional Spaces
-> Works well even with more features than samples

### 2. Robust to Overfitting
-> Especially in high-dimensional space

### 3. Excels on Linearly Separable Data
-> Perfect separation when possible

### 4. Works with Weakly Separable Data
-> Soft margin option handles overlapping classes

### 5. Memory Efficient
-> Only stores support vectors

---

## Limitations of SVM

### 1. Slow for Large Datasets
-> Training time increases significantly with data size

### 2. Sensitive to Noise
-> Performance degrades with noisy, overlapping classes

### 3. Sensitive to Kernel Choice
-> Non-trivial to determine best kernel

### 4. Sensitive to Regularization Parameters
-> C parameter and kernel parameters need tuning

### 5. No Direct Probability Estimates
-> Requires additional computation for probability outputs

---

## Summary:
-> SVM = supervised learning for classification and regression
-> Finds hyperplane that maximizes margin between classes
-> Support vectors = closest points defining the margin
-> Soft margin allows misclassifications (controlled by C)
-> Kernel trick maps data to higher dimensions for nonlinear separation
-> Kernels: Linear, Polynomial, RBF, Sigmoid
-> SVR for regression uses epsilon tube
-> Advantages: High-dimensional, robust to overfitting
-> Limitations: Slow on large data, sensitive to parameters

---

# Supervised Learning with KNN

## What is KNN (K-Nearest Neighbors)?
-> Supervised machine learning algorithm
-> Uses **labeled data points** to learn how to label other data points
-> Used for **both classification and regression**
-> Based on the paradigm: points close to each other should have similar features

### Key Idea:
-> For each query point, find its **nearest neighbors**
-> Predict based on known target labels of those neighbors

---

## How KNN Works

### For Classification:
1. **Pick a value for K** (number of neighbors)
2. **Calculate distance** from query point to all labeled training data
3. **Find K nearest neighbors** (K closest points)
4. **Predict class** using majority vote from K neighbors

### For Regression:
1. Pick a value for K
2. Calculate distance to all training points
3. Find K nearest neighbors
4. Predict using **average or median** of target values

---

## Example: Iris Dataset

### Dataset:
-> 50 samples each from 3 iris species (Setosa, Versicolor, Virginica)
-> 4 features: sepal length, sepal width, petal length, petal width

### Process:
-> For a query point, find K=3 nearest neighbors
-> Majority vote determines the predicted class
-> Example: If 2 neighbors are Virginica and 1 is Versicolor → predict Virginica

### Decision Boundary:
-> K-NN creates regions for each class
-> Points in each region assigned to that class
-> Example: 93% accuracy with K=3

---

## Finding Optimal K

### Method:
1. Test a range of K values using labeled test dataset
2. Calculate prediction accuracy for each K
3. Choose K with **best accuracy**

### Example:
-> Test K=1, K=2, K=3, K=4...
-> Find which K gives highest accuracy
-> In some cases, K=4 might be optimal

---

## KNN as Lazy Learner

### What is Lazy Learner?
-> Does not learn in traditional sense
-> **Stores training data** and makes predictions on-the-fly
-> For each query point:
   1. Calculate distances to all training points
   2. Sort observations by increasing distance
   3. Select top K observations

### Why Still Supervised?
-> Must have labeled training data
-> Uses known labels of neighbors for prediction

---

## Effect of K on Outcome

### If K is Small:
-> Values fluctuate greatly for different query points
-> Model becomes **overfit**
-> Too sensitive to individual neighbors

### If K is Large:
-> Smooths out finer details
-> Model becomes **underfit**
-> Loses local patterns

### Optimal K:
-> Somewhere in between
-> Balances between overfitting and underfitting

---

## Challenges and Solutions

### Challenge 1: Skewed Class Distribution
-> More frequent classes dominate prediction
-> Because they have more neighbors

### Solution:
-> **Weigh classification by distance**
-> Consider distance from test point to each neighbor

### Challenge 2: Features with Large Values
-> Dominate the distance measure
-> Cause biased predictions

### Solution:
-> **Scale features** (standardization)
-> Remove artificial importance

### Challenge 3: Irrelevant Features
-> Like adding noise
-> Requires higher K to avoid overfitting

### Solution:
-> **Keep only relevant features**
-> Improves accuracy and computational efficiency
-> Domain knowledge helps identify relevant features

### How to Check Feature Importance:
1. Train model **with** the feature
2. Train model **without** the feature
3. Compare model performance
4. If performance drops → feature is important

---

## Summary:
-> KNN = supervised algorithm using neighbors to classify/predict
-> Works for both classification (majority vote) and regression (average/median)
-> K = number of nearest neighbors to consider
-> Small K = overfitting, Large K = underfitting
-> Lazy learner: stores data, calculates distances on prediction
-> Scale features to avoid dominance
-> Keep only relevant features for better accuracy

---

# Bias, Variance, and Ensemble Models

## Understanding Bias and Variance

### Analogy: Dart Boards
| Scenario | Bias | Variance | Description |
|----------|------|----------|-------------|
| **Top Left** | Low | Low | High accuracy (near center), grouped together (precise) |
| **Top Right** | Low | High | On-target but spread out |
| **Bottom Left** | High | Low | Off-target but grouped together |
| **Bottom Right** | High | High | Off-target and spread out |

### Key Concepts:
-> **Bias**: How on-target or off-target (accuracy)
-> **Variance**: How spread out (precision)

---

## Prediction Bias

### What is Prediction Bias?
-> Measures average difference between predictions and actual values
-> Average of (predicted value - actual value)

### Formula:
```
Bias = (1/n) × Σ(predicted - actual)
```

### Properties:
-> **Zero bias** = perfect predictor
-> High bias = predictions far from actual values

### Example:
-> Linear model with bias 0.22 vs shifted model with bias 4.22
-> Higher bias = less accurate predictions

---

## Prediction Variance

### What is Prediction Variance?
-> Measures how much predictions **fluctuate** with different training data
-> Model sensitivity to changes in training data

### High Variance:
-> Extremely sensitive to training data
-> **Overfits** training data
-> Tracks noise and outliers
-> Poor generalization to unseen data

### Low Variance:
-> Less sensitive to noise
-> Generalizes well to unseen data
-> More stable predictions

### Example:
-> Same model trained on different subsets
-> Curves align almost perfectly = low variance
-> Curves differ significantly = high variance

---

## Bias-Variance Tradeoff

### Relationship with Model Complexity:

| Complexity | Bias | Variance | Issue |
|------------|------|----------|-------|
| **Low** | High | Low | Underfitting (poor training & test performance) |
| **Optimal** | Medium | Medium | Good generalization |
| **High** | Low | High | Overfitting (good training, poor test performance) |

### Visualization:
```
Error
  ^
  |        \  (Variance)
  |         \
  |          \
  |           \__________  (Bias)
  |                    \
  |                     \
  +-----------------------------> Model Complexity
              ^
         Optimal point
```

### Key Points:
-> As complexity increases: **bias decreases**, **variance increases**
-> Underfitting: High bias → poor predictions
-> Overfitting: High variance → sensitive to training data
-> There is always some irreducible error (random noise)

---

## Weak vs Strong Learners

### Weak Learner:
-> Performs only slightly better than random guessing
-> Characterized by **high bias** and **low variance**
-> Often leads to **underfitting**

### Strong Learner:
-> Performs much better than random
-> Characterized by **low bias** and **high variance**
-> Often leads to **overfitting**

---

## Ensemble Methods

### Purpose:
-> Balance bias and variance
-> Combine multiple models for better predictions

### Common Methods:
1. **Bagging** (Bootstrap Aggregating)
2. **Boosting**

---

## Bagging (Bootstrap Aggregating)

### What is Bagging?
-> Train same model on multiple bootstrap subsets
-> Average predictions across iterations
-> Significantly **reduces prediction variance**

### Process:
1. Create bootstrap samples (random subsets with replacement)
2. Train model on each sample
3. Average predictions

### Result:
-> Reduces variance
-> Lowers overfitting risk
-> Only slightly increases bias

### Example: Random Forests
-> Trains multiple decision trees on bootstrapped data
-> Trees don't need to be deep (shallow)
-> Focus on minimizing prediction bias
-> Aggregation significantly reduces variance

---

## Boosting

### What is Boosting?
-> Ensemble technique building series of weak learners
-> Each learner corrects errors of previous one
-> Systematically reduces prediction error
-> Lowers prediction **bias**

### Process:
1. Train first weak learner
2. Identify misclassified data
3. Increase weights of misclassified data
4. Decrease weights of correctly classified data
5. Train next learner on reweighted data
6. Repeat and form weighted sum

### Popular Boosting Algorithms:
- Gradient Boosting
- XGBoost
- AdaBoost

### Key Points:
-> Sequential training
-> Focuses on correcting mistakes
-> Final model = weighted sum of weak learners

---

## Bagging vs Boosting Comparison

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| **Goal** | Reduce variance | Reduce bias |
| **Training** | Parallel (independent) | Sequential |
| **Base Learners** | High variance, low bias | Low variance, high bias |
| **Data Sampling** | Bootstrap samples | Reweights misclassified |
| **Effect** | Stabilizes predictions | Corrects errors |

---

## Mitigating Bias and Variance

### To Reduce Bias (Underfitting):
-> Increase model complexity
-> Add more features
-> Use boosting

### To Reduce Variance (Overfitting):
-> Reduce model complexity
-> Use more training data
-> Regularization
-> Use bagging (Random Forests)

---

## Summary:
-> Bias = accuracy (how on-target), Variance = precision (how spread out)
-> High bias = underfitting, High variance = overfitting
-> Bias-variance tradeoff: as complexity increases, bias decreases, variance increases
-> Weak learner = high bias, low variance; Strong learner = low bias, high variance
-> Bagging reduces variance, boosting reduces bias
-> Random Forests = bagging with decision trees
-> Boosting builds sequential learners to correct errors
-> Optimal model complexity balances bias and variance

---

# Module Summary

Classification is a supervised machine learning method used to predict labels on new data with applications in churn prediction, customer segmentation, loan default prediction, and multiclass drug prescriptions.

Binary classifiers can be extended to multiclass classification using one-versus-all or one-versus-one strategies.

A decision tree classifies data by testing features at each node, branching based on test results, and assigning classes at leaf nodes.

Decision tree training involves selecting features that best split the data and pruning the tree to avoid overfitting.

Information gain and Gini impurity are used to measure the quality of splits in decision trees.

Regression trees are similar to decision trees but predict continuous values by recursively splitting data to maximize information gain.

Mean Squared Error (MSE) is used to measure split quality in regression trees.

K-Nearest Neighbors (k-NN) is a supervised algorithm used for classification and regression by assigning labels based on the closest labeled data points.

To optimize k-NN, test various k values and measure accuracy, considering class distribution and feature relevance.

Support Vector Machines (SVM) build classifiers by finding a hyperplane that maximizes the margin between two classes, effective in high-dimensional spaces but sensitive to noise and large datasets.

The bias-variance tradeoff affects model accuracy, and methods such as bagging, boosting, and random forests help manage bias and variance to improve model performance.

Random forests use bagging to train multiple decision trees on bootstrapped data, improving accuracy by reducing variance.