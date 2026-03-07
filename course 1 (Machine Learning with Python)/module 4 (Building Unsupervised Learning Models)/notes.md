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

#### When to Use
- **DBSCAN**: When you have a good understanding of your data density and want control over parameters
- **HDBSCAN**: When you want more robust clusters without tuning parameters, especially with varying density data
