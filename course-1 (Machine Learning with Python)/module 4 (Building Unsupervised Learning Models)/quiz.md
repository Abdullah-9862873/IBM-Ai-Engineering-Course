# Module 4 Quiz Answers

## Question 1
**Which clustering method uses a top-down approach?**

- Density-based clustering
- Agglomerative clustering
- **Divisive clustering** ✓
- Partition-based clustering

**Answer**: Divisive clustering

---

## Question 2
**What is the primary objective of the k-means clustering algorithm?**

- Maximize within-cluster variance
- **Minimize within-cluster variance** ✓
- Identify outliers
- Create non-convex clusters

**Answer**: Minimize within-cluster variance

---

## Question 3
**Which of the following scenarios is k-means best suited for?**

- Classifying emails as spam or not spam
- Forecasting stock prices
- **Segmenting customers based on purchasing behavior** ✓
- Detecting anomalies in network traffic

**Answer**: Segmenting customers based on purchasing behavior

---

## Question 4
**Which of the following scenarios is Density-Based Spatial Clustering of Applications with Noise (DBSCAN) best suited for?**

- Classifying text documents by topic
- Predicting the stock market trends
- Segmenting customers based on their age
- **Identifying geographic areas with high crime rates** ✓

**Answer**: Identifying geographic areas with high crime rates

---

## Question 5
**How does Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) improve upon the DBSCAN algorithm?**

- **HDBSCAN can cluster data with varying densities.** ✓
- HDBSCAN does not use density-based clustering.
- HDBSCAN only identifies spherical clusters.
- HDBSCAN requires manual tuning of the number of clusters.

**Answer**: HDBSCAN can cluster data with varying densities.

---

## Question 6
**How might dimension reduction enhance model performance during the clustering process?**

- **Prevents feature loss**
- Increases feature count
- Removes preprocessing steps
- **Simplifies data and improves efficiency** ✓

**Answer**: Simplifies data and improves efficiency

---

## Question 7
**How can clustering facilitate feature selection in a dataset?**

- **Identifies redundant features** ✓
- Eliminates feature engineering
- Treats all features equally
- Increases dataset dimensionality

**Answer**: Identifies redundant features

---

## Question 8
**How does Principal Component Analysis (PCA) contribute to model accuracy in face recognition?**

- Increases training features
- Randomly selects faces
- **Extracts key facial features** ✓
- Ensures equal representation

**Answer**: Extracts key facial features

---

## Question 9
**Which dimensionality reduction algorithm works with complex, high-dimensional data that requires local and global structure preservation for clustering?**

- T-distributed Stochastic Neighbor Embedding (t-SNE)
- **Uniform Manifold Approximation and Projection (UMAP)** ✓
- Principal Component Analysis (PCA)
- Dimensionality reduction is irrelevant when working with complex, high-dimensional data

**Answer**: Uniform Manifold Approximation and Projection (UMAP)

---

## Question 10
**What is the primary purpose of dimensionality reduction algorithms?**

- Increase data set features
- **Simplify data and maintain information content** ✓
- Remove all data noise
- Enhance data complexity

**Answer**: Simplify data and maintain information content

---

## Question 11
**Which of the following is best described as a method in unsupervised learning?**

- Removes dimensionality by discarding half the features at random
- Requires labeled targets to estimate model accuracy
- Converts numeric data into categorical labels for prediction
- **Finds latent structure in unannotated datasets** ✓

**Answer**: Finds latent structure in unannotated datasets

---

## Question 12
**A marketing team wants to use agglomerative hierarchical clustering on delivery routes. Which of the following potential drawbacks should they look for in the large dataset?**

- Specify the number of visitor clusters in advance
- Visualize or interpret the final clusters ineffectively
- **Make the algorithm slow by repeating distance calculations** ✓
- Fit every route into one predefined cluster size

**Answer**: Make the algorithm slow by repeating distance calculations

---

## Question 13
**A fitness app uses K-means clustering to group users based on their daily step count and workout duration. After clustering, the app examines the centroids of each group to understand general activity levels. What does each centroid represent?**

- **The average activity profile of all users in the cluster** ✓
- The sum of squared distances for cluster evaluation
- The maximum activity recorded among cluster members
- The total number of users assigned to the cluster

**Answer**: The average activity profile of all users in the cluster

---

## Question 14
**A bank wants to detect fraudulent transactions by analyzing customer spending patterns using density-based spatial clustering of applications with noise (DBSCAN). Why is DBSCAN a good fit for this analysis?**

- It requires defining a fixed number of clusters to work accurately.
- It forces every transaction into a cluster of suspicious events.
- **It groups data based on density and can isolate suspicious events.** ✓
- It only works with transactions of similar value to generate a cluster.

**Answer**: It groups data based on density and can isolate suspicious events.

---

## Question 15
**Why is customer-behavior data reduced with t-SNE primarily in two-dimensional scatter plots?**

- Automatically label each segment by demographic type
- **Maintains neighborhood similarities, aiding visual discovery of customer segments** ✓
- Enforces linear projections of high-dimensional data
- Allocates equal distance between every pair of reduced points

**Answer**: Maintains neighborhood similarities, aiding visual discovery of customer segments

---

## Question 16
**A marketing team wants to analyze customer purchasing behavior using multiple features like frequency of purchases and product categories. They use principal component analysis (PCA) to reduce dimensionality. What advantages does PCA offer over other techniques?**

- **Eliminates correlations between features**
- Detects and removes outliers automatically
- Requires analysts to guess the number of components
- Transforms features to capture data variance

**Answer**: Eliminates correlations between features (Note: PCA transforms features to be uncorrelated, which helps eliminate correlations)

---

## Question 17
**A research team is working on visualizing the clustering patterns of gene expression data from various samples. They use t-SNE to reduce the dimensionality of the data to two or three dimensions. Why is t-SNE a good choice for this analysis?**

- It reduces the data to a single dimension.
- It finds the global structure of the data.
- It only works with numerical data.
- **It preserves local relationships between similar data points.** ✓

**Answer**: It preserves local relationships between similar data points.
