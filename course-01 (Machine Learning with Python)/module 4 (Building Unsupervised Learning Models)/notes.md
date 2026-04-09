# Module 4: Building Unsupervised Learning Models

## Clustering Strategies in Real-World Applications

### What is Clustering?
- Machine learning technique that automatically groups data points into clusters based on similarities
- Works with unlabeled data, independently finding patterns to form clusters
- Can use one or multiple features to form meaningful clusters

### Applications of Clustering
- **Exploratory Data Analysis**: Uncovers natural groupings (e.g., customer segmentation for targeted marketing)
- **Pattern Recognition**: Groups similar objects, aids in image segmentation (e.g., detecting medical abnormalities)
- **Anomaly Detection**: Identifies outliers to detect fraud or equipment malfunctions
- **Feature Engineering**: Creates new features or reduces dimensionality to improve model performance
- **Data Summarization**: Simplifies data into representative clusters
- **Data Compression**: Replaces data points with cluster centers
- **Feature Identification**: Identifies essential features that distinguish clusters

### Types of Clustering Methods

#### 1. Partition-Based Clustering
- Divides data into non-overlapping groups
- Most common method: **k-means** - identifies k-clusters with minimal variance
- Efficient and scales well with large datasets

#### 2. Density-Based Clustering
- Creates clusters of any shape
- Suitable for irregular clusters and noisy datasets
- Example: **DBSCAN** algorithm

#### 3. Hierarchical Clustering
- Organizes data into a tree of nested clusters
- Generates a **dendrogram** revealing relationships between clusters
- Two main algorithms:
  - **Agglomerative**: Bottom-up approach - merges clusters
  - **Divisive**: Top-down approach - splits clusters

### Comparison: Partition-Based vs Density-Based
- Partition-based clustering struggles with irregular shapes (e.g., interlocking half-circles)
- Density-based clustering can handle irregular shapes but may create unnecessary small clusters

### Hierarchical Clustering Algorithms

#### Agglomerative Hierarchical Clustering (Bottom-Up)
1. Select a distance metric (e.g., distance between centroids)
2. Initialize N clusters (each cluster contains one data point)
3. Compute distance matrix (n-x-n matrix showing distances between each pair)
4. Repeat until desired clusters or merge all into one:
   - Merge two closest clusters
   - Update proximity matrix

#### Divisive Hierarchical Clustering (Top-Down)
1. Start with entire dataset as one cluster
2. Partition into smaller clusters based on similarities/dissimilarities
3. Continue splitting until stopping criterion (e.g., minimum cluster size) is reached

### Key Takeaways
- Clustering is unsupervised learning that finds patterns in unlabeled data
- K-means is a popular partition-based method for customer segmentation
- Density-based methods handle irregular clusters better
- Hierarchical clustering produces dendrograms showing cluster relationships
- Agglomerative uses bottom-up merging; divisive uses top-down splitting

---

## K-Means Clustering

### What is K-Means?
- Iterative, centroid-based clustering algorithm
- Partitions dataset into similar groups based on distance between centroids
- Divides data into k non-overlapping clusters (k is a chosen parameter)
- Constructs clusters with minimal variances around centroids and maximum dissimilarity between clusters

### Key Concepts
- **Centroid**: The average position of all points in the cluster (marked at center)
- **K Value**: Number of clusters
  - Higher k = smaller clusters with more detail
  - Lower k = larger clusters with less detail

### How K-Means Works

#### Step 1: Initialize
- Choose number of clusters (k)
- Randomly select k starting centroid locations (can be data points or other points)

#### Step 2: Assign Points to Clusters (Iterate)
- Compute distance matrix (distances from each point to each centroid)
- Assign each data point to the cluster with the nearest centroid

#### Step 3: Update Centroids
- Calculate new centroid as mean of cluster's data points

#### Step 4: Repeat
- Repeat until centroids stabilize or maximum iterations reached
- Algorithm converges when centroids stop moving

### K-Means Limitations
- **Imbalanced Clusters**: Doesn't perform well when cluster sizes differ significantly
  - Smaller cluster centroid drifts toward larger cluster, consuming its points
- **Non-Convex Clusters**: Assumes clusters are convex (line between two points stays within cluster)
- **Equal Cluster Size**: Assumes clusters contain approximately same number of points
- **Sensitive to Outliers**: Statistical variance is sensitive to outliers

### K-Means Objective
- Minimize within-cluster variance for all clusters simultaneously
- Mathematical formula: Double sum over each cluster (i) and each point (x) within each cluster of the squared distance between x and the cluster's centroid (μi)

### Determining Optimal K

#### Methods:
1. **Silhouette Analysis**: Measures how similar a data point is to its cluster (cohesion) compared to other clusters (separation)
2. **Elbow Method**: Plot of K-Means objective function for different numbers of clusters
3. **Davies-Bouldin Index**: Measures each cluster's average similarity ratio

#### Tips:
- Consider scatter plots between pairs of variables to check for separability
- When standard deviation of blobs increases, cluster centroids get closer together
- If K is too large, K-Means returns unacceptable results

---

## DBSCAN and HDBSCAN Clustering

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

#### What is DBSCAN?
- Density-based spatial clustering algorithm
- Creates clusters with a user-specified density value positioned around a spatial centroid
- Defines neighborhoods around centroids with specified density

#### Key Features
- Discovers clusters of **any shape, size, or density**
- Distinguishes between data points that are part of a cluster vs. noise
- Especially useful for:
  - Datasets with noise or outliers
  - Unknown number of clusters

#### DBSCAN Parameters
- **Epsilon (ε)**: Radius of each neighborhood
- **MinPts (n)**: Minimum number of points required in a neighborhood

#### Point Types in DBSCAN
1. **Core Point**: Has at least n points (including itself) within its epsilon radius
2. **Border Point**: Falls within neighborhood of a core point but doesn't have enough neighbors to be a core point
3. **Noise Point**: Isolated from all core point neighborhoods

#### How DBSCAN Works
1. Select parameters: minimum points (n) and radius (epsilon)
2. For each point, classify as core, border, or noise
3. Grow clusters from core points by including their neighbors
4. Border points are assigned to same cluster as their associated core points
5. Unassigned points are labeled as noise
6. **Not iterative** - grows clusters in one pass without updating

#### Limitations of Centroid-Based Clustering (K-Means)
- Produces spherical/convex shapes only
- Assigns every point to a cluster even if it doesn't fit properly
- Real-world data rarely has simple, spherical patterns

#### DBSCAN Advantages
- Handles arbitrary shapes and shapes within shapes
- Identifies and labels noise/outliers
- No need to specify number of clusters upfront

---

### HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)

#### What is HDBSCAN?
- Variant of DBSCAN that doesn't require parameters to be set
- More flexible and less sensitive to noise and outliers
- Uses **cluster stability** to find optimal clusters

#### Cluster Stability
- Refers to a cluster's ability to not change much when neighborhood size is adjusted within a reasonable range of radii
- Measures persistence of a cluster over a range of distance thresholds

#### How HDBSCAN Works
1. Starts by identifying each point as its own cluster (effectively noise)
2. Progressively agglomerates clusters by incrementally lowering the density threshold
3. Constructs a hierarchical tree
4. Simplifies into a condensed tree, keeping only the most stable clusters across density levels

#### DBSCAN vs HDBSCAN

| Feature | DBSCAN | HDBSCAN |
|---------|--------|---------|
| Parameters | Requires epsilon and minPts | Minimal parameters needed |
| Sensitivity | More sensitive to parameter choices | Less sensitive to noise |
| Density handling | Fixed neighborhood size | Adaptively adjusts neighborhood size |
| Cluster detection | May lump dense regions together | Finds more distinct clusters |
| Output | Flat clusters | Hierarchical structure |

---

## Clustering, Dimension Reduction, and Feature Engineering

### Overview
- Complementary techniques in machine learning and data science
- Work together to improve model performance, quality, and interpretability

### How They Work Together
- **Clustering** helps with feature selection and creation while supporting dimension reduction
- **Dimension reduction** enhances computational efficiency and scalability
- Simplifies visualization of high-dimensional clustering
- Reduces the number of features required for a data model

### Dimension Reduction

#### What is Dimension Reduction?
- Simplifies data structure and improves outcomes
- Commonly used as a pre-processing step for clustering
- Reduces the number of features while preserving key information

#### Challenges with High-Dimensional Data
- Volume expands rapidly as dimensionality increases
- Data points become sparse and less similar
- Distance-based clustering algorithms (k-means, DBSCAN) struggle with high dimensions
- Smaller clusters require more data to fill gaps

#### Common Techniques
- **PCA (Principal Component Analysis)**: Projects data onto principal components
- **t-SNE**: Non-linear dimensionality reduction for visualization
- **UMAP**: Preserves both local and global structure

### Application: Face Recognition with Eigenfaces

#### Process
1. Perform PCA on unlabeled face dataset
2. Extract top eigenfaces (e.g., 150 eigenfaces from 966 faces)
3. Eigenfaces form an orthonormal basis for the feature space
4. Project input data onto eigenface basis
5. Train classifier (e.g., SVM) on projected features

#### Benefits
- Preserves key features for identifying faces
- Minimizes computational load
- Accurately predicts faces with reduced dimensionality

### Clustering for Visualization

#### Problem
- Clustering results cannot be visualized directly in high dimensions (beyond 3D)

#### Solution
- Use dimension reduction to project clusters into 2-3 dimensions
- Methods: PCA, t-SNE, UMAP
- Creates scatter plots to visualize clustering quality
- Enhances cluster interoperability

### Clustering for Feature Selection

#### Concept
- Cluster similar or correlated features
- Identify sets that provide redundant information
- Select representative feature from each cluster
- Reduces total features while preserving valuable information

#### Benefits
- Feature selection (part of feature engineering)
- Can be viewed as dimension reduction
- Helps identify subgroups in data for predictive modeling

### Feature Selection Using K-Means

#### Example
- Features generated with random normal distributions
- Different mean values (1, 5, 10) and variances
- Running k-means on features (not data values) with k=3

#### Results
- Features 1-3 clustered together (redundant features)
- Features 4 and 5 in separate clusters
- Select only one feature from redundant clusters for modeling

### Key Takeaways
- Dimension reduction is essential preprocessing for clustering in high-dimensional spaces
- PCA, t-SNE, and UMAP are standard techniques for reducing dimensions
- Clustering features enables intelligent feature selection
- These techniques enhance model performance, quality, and interpretability

---

## Dimension Reduction Algorithms

### What Are Dimension Reduction Algorithms?
- Reduce the number of dataset features without sacrificing critical dataset information
- Simplify high-dimensional data for analysis and visualization
- Transform original dimensions to create new features
- Essential preprocessing for machine learning models

### Types of Dimension Reduction Algorithms

#### 1. Principal Component Analysis (PCA)

##### Overview
- Linear dimensionality reduction algorithm
- Assumes dataset features are linearly correlated
- Simplifies data, reduces dimensionality, and reduces noise while minimizing information loss

##### How PCA Works
- Transforms features into new uncorrelated variables called principal components
- Retains as much variance as possible
- Principal components are orthogonal to each other
- Components are organized in decreasing order of importance
- First few components contain most information; rest tend to represent noise
- Defines a new coordinate system for the feature space

##### Best For
- Linearly correlated data
- Data with clear principal directions of variance

#### 2. T-Distributed Stochastic Neighbor Embedding (t-SNE)

##### Overview
- Non-linear dimensionality reduction algorithm
- Maps high-dimensional data points to lower-dimensional space (2-3 dimensions)
- Good at finding clusters in complex, high-dimensional data

##### How t-SNE Works
- Focuses on preserving similarity of close points
- Less emphasis on distant points
- Similarity measured as proximity using distance between pairs of points

##### Limitations
- Doesn't scale well with large datasets
- Difficult to tune (sensitive to hyperparameters)

##### Best For
- Data like images and text
- Visualization of clusters in complex data

#### 3. Uniform Manifold Approximation and Projection (UMAP)

##### Overview
- Non-linear dimensionality reduction algorithm
- Alternative to t-SNE
- Based on manifold theory (data lies on lower-dimensional manifold embedded in higher-dimensional space)

##### How UMAP Works
- Constructs high-dimensional graph representation of data
- Optimizes low-dimensional graph structure
- Preserves relationships between points in original data

##### Advantages over t-SNE
- Scales better than t-SNE
- Preserves global structure (not just local)
- Often provides higher clustering performance

##### Best For
- Complex data with both local and global structure
- Large datasets

### Comparison: PCA vs t-SNE vs UMAP

| Algorithm | Type | Preserves | Best For | Limitations |
|-----------|------|-----------|----------|-------------|
| PCA | Linear | Global variance | Linearly correlated data | Assumes linear relationships |
| t-SNE | Non-linear | Local similarity | Visualization, clusters | Doesn't scale well, sensitive to hyperparameters |
| UMAP | Non-linear | Local and global structure | Large datasets, clusters | May not preserve all global structure |

### Practical Example: MakeBlobs Data

#### Input Data
- 3D data with 4 clusters (blobs)
- Some overlap between yellow and purple clusters
- Other blobs distinctly separated

#### PCA Results
- Separates blobs effectively
- Works well when blobs are linearly correlated (differences in means and variances only)

#### t-SNE Results
- Clusters data into four distinct clusters
- Some mixing in overlapping regions (expected)
- Identifies distinct clusters in complex data

#### UMAP Results
- Similar to t-SNE but preserves more global structure
- Yellow and green clusters slightly overlap with purple
- Slightly better than t-SNE for maintaining overall structure

### Key Takeaways
- Dimension reduction algorithms transform high-dimensional data into lower dimensions
- PCA is linear and works best with linearly correlated data
- t-SNE is excellent for visualization but doesn't scale well
- UMAP balances local and global structure preservation with better scalability
- Choice of algorithm depends on data characteristics and use case

---

## Module Summary

- **Clustering** is a machine learning technique used to group data based on similarity, with applications in customer segmentation and anomaly detection.

- **K-means clustering** partitions data into clusters based on the distance between data points and centroids but struggles with imbalanced or non-convex clusters.

- **Heuristic methods** such as silhouette analysis, the elbow method, and the Davies-Bouldin Index help assess k-means performance.

- **DBSCAN** is a density-based algorithm that creates clusters based on density and works well with natural, irregular patterns.

- **HDBSCAN** is a variant of DBSCAN that does not require parameters and uses cluster stability to find clusters.

- **Hierarchical clustering** can be divisive (top-down) or agglomerative (bottom-up) and produces a dendrogram to visualize the cluster hierarchy.

- **Dimension reduction** simplifies data structure, improves clustering outcomes, and is useful in tasks such as face recognition (using eigenfaces).

- **Clustering and dimension reduction** work together to improve model performance by reducing noise and simplifying feature selection.

- **PCA**, a linear dimensionality reduction method, minimizes information loss while reducing dimensionality and noise in data.

- **t-SNE and UMAP** are other dimensionality reduction techniques that map high-dimensional data into lower-dimensional spaces for visualization and analysis.
