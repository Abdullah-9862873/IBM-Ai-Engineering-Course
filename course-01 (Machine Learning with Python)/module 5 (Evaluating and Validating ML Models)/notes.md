# Module 5: Evaluating and Validating ML Models

## Classification Metrics and Evaluation Techniques

### Supervised Learning Evaluation

#### What is Supervised Learning Evaluation?
- Establishes how well a machine learning model can predict the outcome for unseen data
- Involves comparing model predictions to ground truth labels
- Essential during both training and testing phases
- After training, the model is evaluated to estimate how well it can generalize to unseen data

### Train-Test-Split Technique

#### Purpose
- Used to estimate the performance of machine learning algorithms when making predictions
- Prevents overfitting by keeping test data separate from training data

#### How It Works
- Dataset is split into two parts: training set and test set
- Training subset: ~70-80% of the data, used to train the model
- Test subset: Used to evaluate how well the model generalizes to new unseen data

#### Application in Classification
- Model predicts categorical labels
- Assesses how well predictions align with actual labels

---

### Classification Metrics

#### 1. Accuracy

**Definition**: The ratio of correctly predicted instances to the total number of instances in the dataset

**Formula**:
```
Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)
```

**Example**: If 7 out of 10 predictions are correct, accuracy = 70%

#### 2. Confusion Matrix

**Definition**: A table that breaks down the number of ground truth instances of a specific class against the number of predicted class instances

**Components**:
- **True Positive (TP)**: Predicted positive, and it was actually positive
- **True Negative (TN)**: Predicted negative, and it was actually negative
- **False Positive (FP)**: Predicted positive, but actually negative (Type I Error)
- **False Negative (FN)**: Predicted negative, but actually positive (Type II Error)

**Visual Representation**:
- Y-axis: True labels
- X-axis: Predicted labels
- Diagonal entries: Correct predictions
- Off-diagonal entries: Misclassifications

#### 3. Precision

**Definition**: Measures how many of the predicted positive instances are actually positive

**Formula**:
```
Precision = True Positives / (True Positives + False Positives)
```

**When to Use**: When the cost of false positives is high
- Example: Movie recommendation engine - promoting wrong movie to user costs additional with no benefit

#### 4. Recall (Sensitivity)

**Definition**: Measures how many of the actual positive instances are correctly predicted

**Formula**:
```
Recall = True Positives / (True Positives + False Negatives)
```

**When to Use**: When the cost of false negatives is high
- Example: Medical field - missing a patient with illness can have serious consequences

#### 5. F1 Score

**Definition**: Combines precision and recall to represent a model's accuracy using the harmonic mean

**Formula**:
```
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

**When to Use**: When precision and recall are equally important
- Example: Medical diagnosis - both false positives and false negatives have serious consequences

---

### Interpreting Classification Results

#### Using Confusion Matrix
- **Diagonal entries**: Predictions the classifier got right (high values = good)
- **Off-diagonal entries**: Misclassifications
- Heat maps can visualize the confusion matrix with colors representing counts

#### Using Metrics Table
- Calculate precision, recall, and F1 score for each class
- Compare performance across different classes
- Weighted average: Metrics weighted by the support (number of instances) of each class

#### Decision Boundary Visualization
- Shows how the classifier separates different classes
- Background colors distinguish prediction regions
- Dots show actual class distribution

---

### Key Takeaways
- Train-test-split is essential for evaluating model generalization
- Accuracy alone can be misleading for imbalanced datasets
- Confusion matrix provides detailed breakdown of predictions
- Precision important when false positives are costly
- Recall important when false negatives are costly
- F1 score balances precision and recall when both are equally important

---

## Regression Metrics and Evaluation Techniques

### Evaluating Regression Models

#### Why Evaluate Regression Models?
- Regression models are not foolproof and often make prediction errors
- Evaluating a regression model involves determining how accurately the model can predict continuous numerical values
- Example: Predicting exam grades based on midterm scores

#### Understanding Errors
- **Error**: The difference between the predicted values and the actual values
- In linear regression, error is the measure of difference between data points and the trend line
- Multiple ways to determine error when there are multiple data points

#### Why Regression Metrics?
- Provide insight into model's performance
- Show accuracy, error distribution, and error magnitude

---

### Essential Regression Metrics

#### 1. Mean Absolute Error (MAE)

**Definition**: The average absolute difference between the values fitted by the model and the observed historical data

**Formula**:
```
MAE = (1/n) × Σ|y_actual - y_predicted|
```

**Characteristics**:
- Uses absolute values, so all errors are weighted equally
- Less sensitive to outliers compared to MSE

#### 2. Mean Squared Error (MSE)

**Definition**: The sum of the squared difference between the values fitted by the model and observed values divided by the number of historical points minus the number of parameters

**Formula**:
```
MSE = (1/n) × Σ(y_actual - y_predicted)²
```

**Characteristics**:
- Penalizes larger errors more heavily
- Sensitive to outliers

#### 3. Root Mean Squared Error (RMSE)

**Definition**: The square root of the MSE

**Formula**:
```
RMSE = √MSE
```

**Characteristics**:
- Has the same units as the target variable
- Easier to interpret than MSE
- Popular evaluation metric

#### 4. R-squared (R²)

**Definition**: The amount of variance in the dependent variable that the independent variable can explain

**Also Called**: Coefficient of determination

**Formula**:
```
R² = 1 - (Unexplained Variance / Total Variance)
```

**Value Range**:
- 0: Badly fit model
- 1: Perfect model
- Values between 0 and 1 expected in real-world scenarios
- Negative R²: Model performs worse than predicting the mean

**Characteristics**:
- Measures model's goodness of fit
- Easy to understand: "Model explains X% of variation in outcome"
- Assumes target is linearly related to input features
- Can be misleading for nonlinear models

---

### Explained Variance

#### Definition
The sum of squared differences between the predictions and the average value of the actual target data

#### For Perfect Predictor
- Explained variance = Total variance
- Unexplained variance = 0
- R² = 1

#### For Mean Value Model (Constant Prediction)
- Explains no variance
- Explained variance = 0
- R² = 0

---

### Comparing Regression Metrics

#### When to Use Each Metric

| Metric | Use Case |
|--------|----------|
| MAE | When all errors should be weighted equally |
| MSE | When larger errors should be penalized more |
| RMSE | When interpretability with original units is needed |
| R² | When needing to communicate model performance to non-technical audiences |

#### Important Considerations
- No single metric gives universally good measure of goodness of fit
- Visualize results by plotting actual vs predicted values
- Consider transformations (Box-Cox, log) to improve model fit
- All metrics should improve consistently when model fits better

---

### Key Takeaways
- Regression metrics quantify prediction errors
- MAE, MSE, RMSE, and R² are essential regression metrics
- R² measures proportion of variance explained by the model
- RMSE provides interpretable error in original units
- Always visualize results alongside quantitative metrics

---

## Evaluating Unsupervised Learning Models: Heuristics and Techniques

### Challenges in Evaluating Unsupervised Learning

#### Unique Challenges
- No predefined labels or ground truths for training
- Aims to discover hidden patterns and structures in data
- Results are often subjective, requiring careful evaluation for consistency

#### Importance of Stability
- Ensures model performs similarly across varied data subsets or perturbations
- Stable clustering model produces similar clusters despite changes in the dataset

#### Evaluation Approach
- No one-size-fits-all approach
- Combination of methods is essential:
  - Heuristics
  - Domain expertise
  - Metrics
  - Ground truth comparisons
  - Visualization tools

---

### Clustering Evaluation Methods

#### 1. Internal Evaluation Metrics
- Rely on input data only
- Assess clustering quality based on the data itself

#### 2. External Evaluation Metrics
- Use ground truth labels when available
- Compare cluster labels with known classes

#### 3. Generalizability/Stability Evaluation
- Assesses cluster consistency across data variations

#### 4. Dimensionality Reduction for Visualization
- Projects clustering outcomes into 2-3 dimensions
- Enables scatter plots for visualizing clustering quality

#### 5. Cluster-Assisted Learning
- Refines clusters through supervised learning evaluations

#### 6. Domain Expertise
- Provides feedback and interprets clustering results

---

### Internal Clustering Evaluation Metrics

#### 1. Silhouette Score

**Definition**: Compares cohesion within each cluster to separation from others

**Range**: -1 to 1
- Higher values indicate better-defined clusters
- 1: Perfect clustering
- 0: Overlapping clusters
- Negative: Misassigned points

**Formula Components**:
- Distance to nearest neighboring cluster
- Average distance to other points in the same cluster

**Interpretation**:
- > 0.7: Strong clustering
- > 0.5: Reasonable clustering
- < 0.25: Poor clustering

#### 2. Davies-Bouldin Index

**Definition**: Measures average ratio of cluster's compactness to separation from nearest cluster

**Range**: Lower is better
- < 0.3: Excellent clustering
- < 0.6: Reasonable clustering
- Higher values indicate less distinct clusters

#### 3. Inertia (K-Means)

**Definition**: Sum of variances within each cluster

**Characteristics**:
- Lower values suggest more compact clusters
- Tradeoff: Increasing number of clusters reduces variance
- Used with elbow method to find optimal k

---

### External Clustering Evaluation Metrics

Used when labeled or ground-truth data is available

#### 1. Adjusted Rand Index (ARI)

**Definition**: Measures similarity between true labels and clustering outcomes

**Range**: -1 to 1
- 1: Perfect alignment
- 0: Random clustering
- Negative: Worse than random

#### 2. Normalized Mutual Information (NMI)

**Definition**: Quantifies shared information between predicted clusters and true labels

**Range**: 0 to 1
- 1: Perfect agreement
- 0: No shared information

#### 3. Fowlkes-Mallows Index (FM)

**Definition**: Geometric mean of precision and recall based on clustering and label assignments

**Range**: 0 to 1
- Higher score indicates better clustering performance

---

### Evaluating Dimensionality Reduction

#### 1. Explained Variance Ratio (PCA)

**Definition**: Measures variance captured by principal components

**Purpose**: Determines how many components needed for acceptable cumulative explained variance

**Interpretation**:
- First few components often capture most variance
- Additional components may add minimal information

#### 2. Reconstruction Error

**Definition**: Assesses how accurately original data can be reconstructed from reduced representation

**Characteristics**:
- Lower values indicate better information preservation
- Used to evaluate PCA and autoencoders

#### 3. Neighborhood Preservation

**Definition**: Evaluates how well relationships between data points in high-dimensional space are maintained in lower dimensions

**Importance**: Especially for manifold learning algorithms like t-SNE and UMAP

---

### Visualization Examples

#### PCA Visualization
- Projects high-dimensional data to 2-3 dimensions
- First two components often capture most variance
- Enables visualization of class separation

#### Silhouette Plot
- Each bar represents silhouette coefficients for points in each cluster
- Vertical dashed line indicates average silhouette score
- Wide bars indicate well-defined clusters

---

### Key Takeaways
- Unsupervised evaluation requires multiple methods
- Internal metrics assess cluster quality without ground truth
- External metrics compare clustering to known labels
- Stability ensures consistent results across data variations
- Dimensionality reduction evaluation focuses on information preservation
- Visualization tools essential for interpreting unsupervised learning results

---

## Cross-Validation and Advanced Model Validation Techniques

### What is Model Validation?

#### Definition
- Process of doing your best to optimize model without jeopardizing its ability to predict well on unseen data
- Helps prevent overfitting when selecting the best model configuration by tuning hyperparameters
- Ensures model generalizes to new data

### The Problem with Simple Train-Test Split

#### Basic Process
1. Split dataset into training set (~70-80%) and test set (~20-30%)
2. Train model on training set
3. Evaluate model on test set to estimate generalization performance

#### Problem: Data Snooping
- Trying different hyperparameters and choosing the one that performs best on testing data
- Effectively fits the model to the testing data, not training data
- Results in overfitting and poor generalization
- Called "data leakage"

### Three-Part Data Splitting

#### Training Set
- Used to train the model
- Includes optimizing hyperparameters

#### Validation Set
- Subset of training data used during model optimization
- Evaluates model's performance during tuning
- Helps select best hyperparameters

#### Test Set
- Held back from training
- Unseen data used for final evaluation
- Only used once after model is fully trained

---

### Cross-Validation

#### Purpose
- Decouples model tuning from final evaluation
- Provides more robust estimate of model performance

#### Algorithm
1. Split data into training and testing data
2. Further split training data into training set and validation set
3. Optimize hyperparameters by training on training set and measuring on validation set
4. Choose best hyperparameters
5. Evaluate final model on completely unseen test data

---

### K-Fold Cross-Validation

#### How It Works
1. Divide data into K equal-sized folds
2. For each set of hyperparameters:
   - For each fold:
     - Train model on K-1 folds
     - Test model on the selected fold
     - Store the score
3. Compute aggregated score across all folds
4. Select hyperparameters that led to best model

#### Benefits
- Every data point used for both training and validation
- Greatly increases data utilization
- Reduces overfitting to specific validation set
- Provides more robust generalization estimate

#### Typical Values
- K typically 5 to 10

---

### Stratified Cross-Validation

#### When to Use
- Classification problems with imbalanced classes
- Many observations in one class, few in another

#### Purpose
- Ensures class distribution is preserved in each validation fold
- Prevents bias in evaluation process

---

### Handling Skewed Data in Regression

#### Problem
- Target variable highly skewed
- Many models assume normally distributed target

#### Solutions
- Log transformation
- Box-Cox transformation

#### Benefits
- Reduces skewness
- Helps models fit data better
- Improves linear regression performance

---

### Key Takeaways
- Model validation prevents overfitting and ensures generalization
- Data snooping occurs when test data is used during model selection
- Three-part splitting: training, validation, and test sets
- K-fold cross-validation provides robust performance estimates
- Stratified cross-validation preserves class distribution
- Target transformations help with skewed data in regression

---

## Regularization in Linear Regression

### What is Regularization?

#### Definition
- Regression technique to prevent overfitting
- Constrains the model during training
- Discourages overfitting to training data
- Achieves this by suppressing the size of coefficients

#### Modified Cost Function
```
Regularized Cost Function = MSE + λ × Penalty Term
```
- λ (lambda): Parameter controlling penalty influence
- Penalty: Measures size of coefficients

---

### Linear Regression (Ordinary Least Squares)

#### Definition
- Models relationship between variables by fitting a straight line
- Predictions are linear combinations of features
- Goal: Minimize MSE between predicted and actual target values

#### Mathematical Form
```
ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```
- θ₀: Bias/intercept term
- θ₁-θₙ: Coefficients/weights

#### Problem
- Sensitive to noisy data
- Prone to overfitting
- No penalty term to constrain coefficients

---

### Ridge Regression (L2 Regularization)

#### Penalty Term
- L2 or sum-of-squares penalty
- Shrinks coefficients toward zero but not exactly to zero

#### Formula
```
Cost = MSE + λ × Σ(coefficients)²
```

#### Characteristics
- Helps shrink coefficients
- Reduces variance
- Keeps all features but reduces their impact
- Good when all features may be relevant

---

### Lasso Regression (L1 Regularization)

#### Penalty Term
- L1 or sum of absolute values penalty
- Can shrink some coefficients to exactly zero

#### Formula
```
Cost = MSE + λ × Σ|coefficients|
```

#### Characteristics
- Performs feature selection
- Useful for sparse data (few significant features)
- Creates sparse coefficients
- Good for high-dimensional data

---

### Comparing Linear, Ridge, and Lasso

| Method | Penalty | Feature Selection | Best For |
|--------|---------|-------------------|----------|
| Linear | None | No | Low noise, all features relevant |
| Ridge (L2) | Sum of squares | No (shrinks but keeps all) | When all features may be relevant |
| Lasso (L1) | Sum of absolute values | Yes (can set to zero) | Sparse data, feature selection |

---

### Performance in Different Scenarios

#### High Signal-to-Noise Ratio (Low Noise)
- All three methods predict non-zero coefficients well
- Lasso can find zero coefficients exactly
- Linear and ridge have difficulty with zero coefficients

#### Low Signal-to-Noise Ratio (High Noise)
- Linear regression performs very poorly
- Overestimates coefficients
- Assigns large values where they should be zero
- Ridge and lasso significantly outperform linear regression

#### Sparse Coefficients
- Lasso is best at finding zero coefficients
- Ridge and lasso similar for non-zero predictions
- Lasso excellent for feature selection

#### Non-Sparse Coefficients
- All methods predict well
- Ridge slightly outperforms lasso on non-zero coefficients
- Lasso still good at identifying zero coefficients

---

### When to Use Each Method

#### Use Linear Regression When
- Low noise in data
- All features are relevant
- Simple, interpretable model needed

#### Use Ridge Regression When
- Moderate noise
- All features may contribute
- Multicollinearity present (correlated features)

#### Use Lasso Regression When
- High noise (low SNR)
- Sparse coefficients (few important features)
- Feature selection needed
- Data compression tasks

---

### Key Takeaways
- Regularization prevents overfitting by constraining coefficients
- Ridge (L2) shrinks coefficients but keeps all features
- Lasso (L1) can set coefficients to zero (feature selection)
- Lambda (λ) controls penalty strength - higher = more regularization
- Lasso is best for sparse data and feature selection
- Ridge is best when all features may be relevant

---

## Data Leakage and Other Pitfalls

### What is Data Leakage?

#### Definition
- When model's training data includes information not available in real-world deployment
- Training data contains "future" information that wouldn't be accessible after deployment
- Deceives model, making it perform misleadingly well during training and validation

#### Example: House Price Prediction
- Engineer feature using average of actual home prices over entire dataset
- Model performs well on test data but fails in production
- Deployed model can't access global averages it was trained with

#### Why It's Problematic
- Test dataset also contains leaked data
- Evaluation won't detect poor generalizability until production
- Leads to inaccurate performance expectations

---

### Data Snooping

#### Definition
- When training set contains information about testing set
- Model sees data it shouldn't have access to

#### Common Causes
- Including future information to predict past outcomes
- Engineering features using entire dataset before splitting

---

### Mitigation Strategies

#### Feature Engineering
- Avoid features like global averages or statistics derived from entire dataset
- Use only information that would be available at prediction time

#### Proper Data Splitting
- Ensure separation between training, validation, and test sets
- Avoid overlap or contamination
- No feature should contain unavailable information in production

#### Cross-Validation Best Practices
- Run processing pipelines independently on each training fold
- Apply fitted pipeline to corresponding validation fold
- Important for time-dependent data

#### Time-Series Data
- Use time-series split instead of random train-test split
- Training set always precedes test set
- Ensures temporal order is maintained

---

### Feature Importance Interpretation Pitfalls

#### 1. Correlated Features
- Highly correlated features share importances
- Lowers apparent influence of individual features
- Need to handle multicollinearity

#### 2. Feature Selection Risk
- Selecting most important features can degrade results
- Blind selection may remove features that contribute through interactions

#### 3. Scale Sensitivity
- Some algorithms (linear regression) don't account for feature scale
- Unscaled data can distort importance rankings

#### 4. Correlation vs Causation
- Feature importance indicates correlation, not causation
- Important features don't necessarily drive outcomes

#### 5. Feature Interactions
- Individual importance rankings don't account for interactions
- Two unimportant features together may be crucial
- Example: Linear regression sees both as unimportant, but their product is key

---

### Common Modeling Pitfalls

#### 1. Raw Data Without Transformation
- Using features without appropriate selection or transformation
- Prevents discovering optimal model

#### 2. Wrong Evaluation Metrics
- Choosing inappropriate metric for the problem
- Misinterpreting metrics can mislead evaluation

#### 3. Class Imbalance
- Failing to address imbalanced classification problems
- Biases predictions toward majority classes

#### 4. Blind Reliance on Automation
- AutoML tools are powerful but require understanding
- Must understand data and model the system creates

#### 5. What-If Scenarios Without Causality
- Models without causal features generate invalid what-if scenarios
- Predictions based on hypothetical changes can be misleading
- Causal relationships essential for intervention modeling

---

### Best Practices Summary

1. Carefully select training and testing data
2. Ensure no future information leaks into training data
3. Run data processing pipelines separately for train/test
4. Use appropriate cross-validation for data type
5. Understand feature importance and interactions
6. Choose evaluation metrics appropriate for the problem
7. Address class imbalance in classification
8. Consider causal relationships for intervention predictions

---

### Key Takeaways
- Data leakage occurs when training data includes information not available in production
- Avoid data leakage by ensuring proper data separation and feature engineering
- Cross-validation must be implemented carefully to avoid leakage across folds
- Feature importance shows correlation, not causation
- Watch for correlated features and feature interactions
- Choose evaluation metrics appropriate for your specific problem

---

## Module Summary

- **Supervised learning evaluation** assesses a model's ability to predict outcomes for unseen data, often using a train/test split to estimate performance.

- **Key metrics for classification evaluation** include accuracy, confusion matrix, precision, recall, and the F1 score, which balances precision and recall.

- **Regression model evaluation metrics** include MAE, MSE, RMSE, R-squared, and explained variance to measure prediction accuracy.

- **Unsupervised learning models** are evaluated for pattern quality and consistency using metrics like Silhouette Score, Davies-Bouldin Index, and Adjusted Rand Index.

- **Dimensionality reduction evaluation** involves Explained Variance Ratio, Reconstruction Error, and Neighborhood Preservation to assess data structure retention.

- **Model validation**, including dividing data into training, validation, and test sets, helps prevent overfitting by tuning hyperparameters carefully.

- **Cross-validation methods**, especially K-fold and stratified cross-validation, support robust model validation without overfitting to test data.

- **Regularization techniques**, such as ridge (L2) and lasso (L1) regression, help prevent overfitting by adding penalty terms to linear regression models.

- **Data leakage** occurs when training data includes information unavailable in real-world data, which is preventable by separating data properly and mindful feature selection.

- **Common modelling pitfalls** include misinterpreting feature importance, ignoring class imbalance, and making causal inferences without sufficient evidence.

- **Feature importance assessments** should consider redundancy, scale sensitivity, and avoid misinterpretation, as well as inappropriate assumptions about causation.

