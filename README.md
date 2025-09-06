# K-means vs K-medoids Clustering Analysis

Implementation and comparison of K-means vs K-medoids clustering algorithms for both vector and matrix data with performance analysis and visualization.

## Overview

This project provides a comprehensive implementation of both K-means and K-medoids clustering algorithms, designed to work with both traditional vector data and matrix data. The system includes performance evaluation metrics, visualization tools, and comparative analysis to determine which algorithm performs better under different data conditions.

## Features

- **Dual Algorithm Support**: Complete implementations of both K-means and K-medoids
- **Vector & Matrix Clustering**: Supports clustering of both vector data and matrix data
- **Performance Evaluation**: Intra-cluster variance calculation for algorithm comparison
- **Visualization**: Scatter plot generation for cluster analysis
- **Flexible Distance Metrics**: Euclidean distance for vectors, matrix norms for matrices
- **Comparative Analysis**: Side-by-side algorithm performance evaluation

## Algorithm Implementations

### Vector Clustering Functions

#### `initialize_centroids(vertices, num_clusters)`
Randomly selects initial centroids for clustering initialization.
- **Input**: Data points and desired number of clusters
- **Output**: Initial centroid positions

#### `assign_clusters(vertices, centroids)`
Assigns each data point to the nearest centroid based on Euclidean distance.
- **Method**: Minimum distance assignment
- **Distance Metric**: Euclidean distance

#### `update_centroids(clusters)`
Updates centroids in K-means by calculating the mean position of points in each cluster.
- **Algorithm**: K-means centroid update rule
- **Method**: Average position calculation

#### `update_medoids(clusters)`
Updates medoids in K-medoids by selecting the point that minimizes total intra-cluster distance.
- **Algorithm**: K-medoids medoid selection
- **Method**: Minimize sum of distances within cluster

#### `iterate_clustering(vertices, num_clusters, iterations, clustering_type)`
Main clustering iteration function supporting both K-means and K-medoids.
- **Iterations**: 50 iterations for convergence
- **Support**: Both algorithm types

### Matrix Clustering Functions

#### `matrix_norm_distance(matrix1, matrix2, norm_type='fro')`
Calculates matrix norm distance between two matrices.
- **Default**: Frobenius norm
- **Purpose**: Distance metric for matrix clustering

#### `assign_matrix_clusters(matrices, centroids, norm_type='fro')`
Assigns matrices to clusters based on matrix norm distances.
- **Extension**: Matrix version of standard cluster assignment

#### `update_matrix_centroids(clusters)` & `update_matrix_medoids(clusters, old_medoids, norm_type='fro')`
Update functions specifically designed for matrix clustering.
- **Matrix Operations**: Element-wise operations on matrices
- **Norm Types**: Flexible norm selection

#### `iterate_matrix_clustering(matrices, num_clusters, iterations, clustering_type, norm_type='fro')`
Complete matrix clustering iteration function.
- **Matrix Support**: Full clustering pipeline for matrices

### Visualization & Analysis

#### `plot_clusters(final_clusters, title, ax, xlabel, ylabel)`
Creates scatter plot visualizations of clustering results.
- **Color Coding**: Each cluster displayed in different colors
- **Customizable**: Flexible axis labels and titles

#### `intra_cluster_variance(clusters, centers)`
Calculates average squared distance between points and their cluster centers.
- **Metric**: Primary performance evaluation metric
- **Lower = Better**: Lower variance indicates tighter clusters

#### `print_final_centers(centers, name)`
Displays the final centroids or medoids for each cluster.
- **Output**: Formatted center coordinates

## Performance Results

### Example Performance Comparisons

```
Dataset 1:
K-means Intra-Cluster Variance: 2558.78
K-medoids Intra-Cluster Variance: 5085.90

Dataset 2 (Vectors X):
K-means: 0.0361, K-medoids: 0.0463

Dataset 3 (Vectors Y):  
K-means: 0.771, K-medoids: 1.470

Matrix Dataset A:
K-means: 0.583, K-medoids: 0.800

Matrix Dataset B:
K-means: 19.25, K-medoids: 24.13
```

### Key Findings

1. **General Performance**: K-means consistently achieves lower intra-cluster variance
2. **Vector Data**: Similar performance when data is uniformly distributed
3. **Scattered Data**: K-means excels when few points are scattered far from clusters  
4. **Matrix Clustering**: K-means shows better performance on complex matrix datasets
5. **Dataset Dependency**: Performance gap varies significantly by data characteristics

## Usage

### Basic Vector Clustering

```python
# Initialize and run K-means
vertices = your_data_points
num_clusters = 3
iterations = 50

# K-means clustering
kmeans_clusters, kmeans_centers = iterate_clustering(
    vertices, num_clusters, iterations, "k-means"
)

# K-medoids clustering  
kmedoids_clusters, kmedoids_centers = iterate_clustering(
    vertices, num_clusters, iterations, "k-medoids"
)

# Performance comparison
kmeans_variance = intra_cluster_variance(kmeans_clusters, kmeans_centers)
kmedoids_variance = intra_cluster_variance(kmedoids_clusters, kmedoids_centers)
```

### Matrix Clustering

```python
# Matrix clustering
matrices = your_matrix_data
matrix_clusters, matrix_centers = iterate_matrix_clustering(
    matrices, num_clusters, iterations, "k-means", norm_type='fro'
)

# Visualization
plot_clusters_matrices(matrices, matrix_clusters, matrix_centers)
```

## Requirements

```python
numpy
matplotlib
scipy  # for matrix operations
```

## Applications

- **Data Analysis**: Exploratory data analysis and pattern recognition
- **Computer Vision**: Image segmentation and feature clustering
- **Machine Learning**: Data preprocessing and dimensionality analysis  
- **Research**: Algorithm comparison and performance studies
- **Education**: Understanding clustering algorithm differences

## Algorithm Comparison

### K-means Advantages
- **Lower Variance**: Consistently achieves tighter clusters
- **Computational Efficiency**: Faster convergence in most cases
- **Scalability**: Better performance on larger datasets

### K-medoids Advantages  
- **Robustness**: Less sensitive to outliers
- **Interpretability**: Centers are actual data points
- **Distance Metrics**: More flexible with non-Euclidean distances

## Experimental Methodology

1. **Initialization**: Random centroid/medoid selection
2. **Iteration**: 50 iterations for both algorithms
3. **Evaluation**: Intra-cluster variance calculation
4. **Visualization**: Scatter plot generation
5. **Comparison**: Side-by-side performance analysis

## Future Enhancements

- **Additional Algorithms**: DBSCAN, Hierarchical clustering
- **Distance Metrics**: Custom distance functions
- **Optimization**: Convergence detection and early stopping
- **Interactive Visualization**: Real-time clustering exploration
- **Parallel Processing**: Multi-threaded matrix operations
- **Statistical Analysis**: Confidence intervals and significance testing
