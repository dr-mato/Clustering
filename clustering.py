import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

vertices = [
    [2023, 1051],  # Calling (Spider-Man: Across the Spider-Verse) (Metro Boomin & Swae Lee, NAV, feat. A Boogie Wit da H...)
    [2023, 954],   # Cupid - Twin Ver. (FIFTY FIFTY)
    [2023, 875],   # Last Night (Morgan Wallen)
    [2023, 861],   # Calm Down (Rema & Selena Gomez)
    [2023, 790],   # Ella Baila Sola (Eslabon Armado & Peso Pluma)
    [2023, 711],   # Creepin' (Metro Boomin, The Weeknd & 21 Savage)
    [2023, 702],   # Kill Bill (SZA)
    [2023, 684],   # Anti-Hero (Taylor Swift)
    [2023, 620],   # Die For You (Remix) (The Weeknd & Ariana Grande)
    [2022, 981],   # As It Was (Harry Styles)
    [2022, 876],   # I Ain't Worried (OneRepublic)
    [2022, 736],   # I Like You (A Happier Song) (Post Malone feat. Doja Cat)
    [2022, 699],   # I'm Good (Blue) (David Guetta & Bebe Rexha)
    [2022, 676],   # Unholy (Sam Smith & Kim Petras)
    [2022, 655],   # Left and Right (Charlie Puth feat. Jung Kook)
    [2022, 617],   # First Class (Jack Harlow)
    [2021, 742],   # Heat Waves (Glass Animals)
    [2021, 736],   # STAY (The Kid LAROI & Justin Bieber)
    [2021, 667],   # Cold Heart (PNAU Remix) (Elton John & Dua Lipa)
    [2021, 643],   # Shivers (Ed Sheeran)
    [2020, 607],   # Blinding Lights (The Weeknd)
    [2020, 596],   # Watermelon Sugar (Harry Styles)
    [2020, 560],   # Levitating (Dua Lipa feat. DaBaby)
    [2019, 578],   # Someone You Loved (Lewis Capaldi)
    [2019, 563],   # Dance Monkey (Tones And I)
    [2019, 562],   # Memories (Maroon 5)
    [2019, 524],   # Circles (Post Malone)
    [2018, 547],   # Sunflower (Spider-Man: Into the Spider-Verse) (Post Malone & Swae Lee)
    [2018, 497],   # Happier (Marshmello feat. Bastille)
    [2018, 494],   # thank u, next (Ariana Grande)
    [2018, 470],   # Better Now (Post Malone)
    [2017, 482],   # Shape of You (Ed Sheeran)
    [2017, 466],   # Something Just Like This (The Chainsmokers & Coldplay)
    [2017, 448],   # Believer (Imagine Dragons)
    [2017, 434],   # Perfect (Ed Sheeran)
    [2016, 439],   # Starboy (The Weeknd feat. Daft Punk)
    [2016, 435],   # 24K Magic (Bruno Mars)
    [2016, 430],   # Don't Let Me Down (The Chainsmokers feat. Daya)
    [2015, 446],   # Uptown Funk (Mark Ronson feat. Bruno Mars)
    [2015, 429],   # Stitches (Shawn Mendes)
    [2015, 422],   # See You Again (Wiz Khalifa feat. Charlie Puth)
    [2014, 415],   # Thinking Out Loud (Ed Sheeran)
    [2014, 408],   # Take Me to Church (Hozier)
    [2014, 405],   # All of Me (John Legend)
    [2014, 400],   # Shake It Off (Taylor Swift)
    [2013, 387],   # Radioactive (Imagine Dragons)
    [2013, 380],   # Counting Stars (OneRepublic)
    [2013, 377],   # Get Lucky (Daft Punk feat. Pharrell Williams)
    [2013, 370],   # Can't Hold Us (Macklemore & Ryan Lewis feat. Ray Dalton)
    [2012, 360],   # Somebody That I Used to Know (Gotye feat. Kimbra)
    [2012, 359],   # We Are Young (fun. feat. Janelle Monáe)
    [2012, 355],   # Call Me Maybe (Carly Rae Jepsen)
    [2011, 348],   # Rolling in the Deep (Adele)
    [2011, 344],   # Party Rock Anthem (LMFAO feat. Lauren Bennett & GoonRock)
    [2011, 340],   # Give Me Everything (Pitbull feat. Ne-Yo, Afrojack & Nayer)
    [2010, 333],   # Just the Way You Are (Bruno Mars)
    [2010, 328],   # Love the Way You Lie (Eminem feat. Rihanna)
    [2010, 323],   # Dynamite (Taio Cruz)
    [2009, 318],   # I Gotta Feeling (Black Eyed Peas)
    [2009, 310],   # Use Somebody (Kings of Leon)
    [2009, 309],   # Poker Face (Lady Gaga)
]

X = np.random.rand(100, 2)
Y = np.vstack([np.random.rand(90, 2), np.random.rand(10, 2) * 10])
A = np.random.rand(100, 3, 3)
B = np.vstack([np.random.rand(90, 3, 3), np.random.rand(10, 3, 3) * 10])

def initialize_centroids(data, num_clusters): # works for both k-means and k-medoids
    centroids = []
    while len(centroids) != num_clusters:
        new_centroid = data[np.random.choice(len(data))]
        if not any(np.array_equal(new_centroid, centroid) for centroid in centroids):
            centroids.append(new_centroid)
    return centroids

def assign_clusters(data, centroids): # works for both k-means and k-medoids
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [np.linalg.norm(np.array(point) - np.array(centroid)) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        clusters[closest_centroid].append(point)
    return clusters 

def matrix_norm_distance(matrix1, matrix2, norm_type='fro'):
    return np.linalg.norm(matrix1 - matrix2, ord=norm_type)

def assign_matrix_clusters(matrices, centroids, norm_type='fro'): 
    clusters = [[] for _ in range(len(centroids))]
    for matrix in matrices:
        distances = [matrix_norm_distance(matrix, centroid, norm_type) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        clusters[closest_centroid].append(matrix)
    return clusters

def update_centroids(clusters, old_centroids):
    new_centroids = []
    for cluster in clusters:
        if cluster:  # Ensure the cluster is not empty
            new_centroid = np.mean(cluster, axis=0)
            new_centroids.append(new_centroid.tolist())
        else:
            # If a cluster is empty, keep its old centroid
            new_centroids.append(old_centroids[len(new_centroids)])
    return new_centroids

def update_medoids(clusters, old_medoids):
    new_medoids = []
    for cluster in clusters:
        if cluster:  # Ensure the cluster is not empty
            cluster_array = np.array(cluster)
            distances_sum = np.sum(np.linalg.norm(cluster_array[:, None] - cluster_array, axis=2), axis=0)
            new_medoid = cluster_array[np.argmin(distances_sum)]
            new_medoids.append(new_medoid)
        else:
            # If a cluster is empty, keep its old medoid
            new_medoids.append(old_medoids[len(new_medoids)])
    return new_medoids

def update_matrix_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if cluster:  # Ensure the cluster is not empty
            new_centroid = np.mean(cluster, axis=0)
            new_centroids.append(new_centroid)
        else:
            # If a cluster is empty, append None
            new_centroids.append(None)
    return new_centroids

def update_matrix_medoids(clusters, old_medoids, norm_type='fro'):
    new_medoids = []
    for cluster in clusters:
        if cluster:  # Ensure the cluster is not empty
            cluster_array = np.array(cluster)
            distances_sum = np.sum([[matrix_norm_distance(a, b, norm_type) for a in cluster_array] for b in cluster_array], axis=0)
            new_medoid = cluster_array[np.argmin(distances_sum)]
            new_medoids.append(new_medoid)
        else:
            # Keep the old medoid if the cluster is empty
            new_medoids.append(old_medoids[len(new_medoids)])
    return new_medoids

def iterate_clustering(data, num_clusters, iterations, clustering_type="k-means"):
    centroids = initialize_centroids(data, num_clusters)
    medoids = initialize_centroids(data, num_clusters)
    
    for _ in range(iterations):
        if clustering_type == "k-means":
            clusters = assign_clusters(data, centroids)
            old_centroids = centroids[:]
            centroids = update_centroids(clusters, old_centroids)
        elif clustering_type == "k-medoids":
            clusters = assign_clusters(data, medoids)
            old_medoids = medoids[:]
            medoids = update_medoids(clusters, old_medoids)
    
    final_clusters = assign_clusters(data, centroids if clustering_type == "k-means" else medoids)
    final_centers = centroids if clustering_type == "k-means" else medoids
    
    return final_clusters, final_centers

def iterate_matrix_clustering(matrices, num_clusters, iterations, clustering_type="k-means", norm_type='fro'):
    centroids = initialize_centroids(matrices, num_clusters)
    medoids = initialize_centroids(matrices, num_clusters)

    for _ in range(iterations):
        if clustering_type == "k-means":
            clusters = assign_matrix_clusters(matrices, centroids, norm_type)
            centroids = update_matrix_centroids(clusters)
        elif clustering_type == "k-medoids":
            clusters = assign_matrix_clusters(matrices, medoids, norm_type)
            medoids = update_matrix_medoids(clusters, medoids, norm_type)
    
    final_clusters = assign_matrix_clusters(matrices, centroids if clustering_type == "k-means" else medoids, norm_type)
    final_centers = centroids if clustering_type == "k-means" else medoids
    
    return final_clusters, final_centers

def plot_clusters(final_clusters, title, ax, xlabel, ylabel):
    colors = ['red', 'blue', 'green', 'yellow']
    for i, cluster in enumerate(final_clusters):
        cluster_array = np.array(cluster)
        ax.scatter(cluster_array[:, 0], cluster_array[:, 1], color=colors[i], label=f'Cluster {i+1}')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    
def plot_clusters_matrices(matrices, final_clusters, centers):
    # Flatten matrices to a 2D array for visualization using PCA
    flattened_matrices = np.array([matrix.flatten() for matrix in matrices])
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(flattened_matrices)
    
    # Plot the reduced data points
    colors = ['red', 'blue', 'green', 'yellow']
    for i, cluster in enumerate(final_clusters):
        cluster_indices = np.where(np.all(matrices[:, None] == cluster, axis=(2, 3)))[0]
        cluster_points = reduced_data[cluster_indices]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i+1}')
    
    # Plot the centers
    flattened_centers = np.array([center.flatten() for center in centers])
    reduced_centers = pca.transform(flattened_centers)
    plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], color='black', marker='x', s=100, label='Centers')
    
    plt.legend()
    plt.title("Matrix Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()
    
def intra_cluster_variance(clusters, centers):
    total_variance = 0
    for i, cluster in enumerate(clusters):
        if cluster:  # Ensure the cluster is not empty
            cluster_array = np.array(cluster)
            center = np.array(centers[i])
            distances = np.linalg.norm(cluster_array - center, axis=1)
            total_variance += np.sum(distances ** 2)
    return total_variance / len(np.concatenate(clusters))

def intra_cluster_variance_matrices(matrices, clusters, centers):
    total_variance = 0
    cluster_variances = []
    
    for cluster, center in zip(clusters, centers):
        if len(cluster) > 0:  # Ensure the cluster is not empty
            # Calculate variance as sum of squared Frobenius norms
            variance = np.mean([np.linalg.norm(matrix - center, 'fro') ** 2 for matrix in cluster])
            cluster_variances.append(variance)
            total_variance += variance
        
    return total_variance / len(clusters), cluster_variances

def print_final_centers(centers, name):
    print(f"{name} Centers:")
    for i, center in enumerate(centers):
        print(f"Center {i+1}: {center}")
        
def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    num_clusters = 4
    iterations = 50

    final_k_means_clusters, centroids = iterate_clustering(vertices, num_clusters, iterations, "k-means")
    plot_clusters(final_k_means_clusters, "K-means Clustering", axes[0], "Year", "Peak Position")
    print_final_centers(centroids, "K-means")

    final_k_medoids_clusters, medoids = iterate_clustering(vertices, num_clusters, iterations, "k-medoids")
    plot_clusters(final_k_medoids_clusters, "K-medoids Clustering", axes[1], "Year", "Peak Position")
    print_final_centers(medoids, "K-medoids")
    
    plt.tight_layout()
    plt.show()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    final_k_means_clusters_for_X, centroids_for_X = iterate_clustering(X, num_clusters, iterations, "k-means")
    plot_clusters(final_k_means_clusters_for_X, "K-means Clustering (X)", axes[0], "X1", "X2")
    print_final_centers(centroids_for_X, "K-means")

    final_k_medoids_clusters_for_X, medoids_for_X = iterate_clustering(X, num_clusters, iterations, "k-medoids")
    plot_clusters(final_k_medoids_clusters_for_X, "K-medoids Clustering (X)", axes[1], "X1", "X2")
    print_final_centers(medoids_for_X, "K-medoids")
    
    plt.tight_layout()
    plt.show()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    final_k_means_clusters_for_Y, centroids_for_Y = iterate_clustering(Y, num_clusters, iterations, "k-means")
    plot_clusters(final_k_means_clusters_for_Y, "K-means Clustering (Y)", axes[0], "Y1", "Y2")
    print_final_centers(centroids_for_Y, "K-means")

    final_k_medoids_clusters_for_Y, medoids_for_Y = iterate_clustering(Y, num_clusters, iterations, "k-medoids")
    plot_clusters(final_k_medoids_clusters_for_Y, "K-medoids Clustering (X)", axes[1], "Y1", "Y2")
    print_final_centers(medoids_for_Y, "K-medoids")
    
    plt.tight_layout()
    plt.show()

    final_k_means_clusters_for_A, centroids_for_A = iterate_matrix_clustering(A, num_clusters, iterations, "k-means", 'fro')
    plot_clusters_matrices(A, final_k_means_clusters_for_A, centroids_for_A)
    print_final_centers(centroids_for_A, "K-means (Matrices)")

    final_k_medoids_clusters_for_A, medoids_for_A = iterate_matrix_clustering(A, num_clusters, iterations, "k-medoids", 'fro')
    plot_clusters_matrices(A, final_k_medoids_clusters_for_A, medoids_for_A)
    print_final_centers(medoids_for_A, "K-medoids (Matrices)")
    
    final_k_means_clusters_for_B, centroids_for_B = iterate_matrix_clustering(B, num_clusters, iterations, "k-means", 'fro')
    plot_clusters_matrices(B, final_k_means_clusters_for_B, centroids_for_B)
    print_final_centers(centroids_for_B, "K-means (Matrices)")
    
    final_k_medoids_clusters_for_B, medoids_for_B = iterate_matrix_clustering(B, num_clusters, iterations, "k-medoids", 'fro')
    plot_clusters_matrices(B, final_k_medoids_clusters_for_B, medoids_for_B)
    print_final_centers(medoids_for_B, "K-medoids (Matrices)")
    
    intra_cluster_variance_k_means = intra_cluster_variance(final_k_means_clusters, centroids)
    intra_cluster_variance_k_medoids = intra_cluster_variance(final_k_medoids_clusters, medoids)
    print(f"K-means Intra-Cluster Variance: {intra_cluster_variance_k_means}, K-medoids Intra-Cluster Variance: {intra_cluster_variance_k_medoids}")
    intra_cluster_variance_k_means_for_X = intra_cluster_variance(final_k_means_clusters_for_X, centroids_for_X)
    intra_cluster_variance_k_medoids_for_X = intra_cluster_variance(final_k_medoids_clusters_for_X, medoids_for_X)
    print(f"K-means Intra-Cluster Variance (X): {intra_cluster_variance_k_means_for_X}, K-medoids Intra-Cluster Variance (X): {intra_cluster_variance_k_medoids_for_X}")
    intra_cluster_variance_k_means_for_Y = intra_cluster_variance(final_k_means_clusters_for_Y, centroids_for_Y)
    intra_cluster_variance_k_medoids_for_Y = intra_cluster_variance(final_k_medoids_clusters_for_Y, medoids_for_Y)
    print(f"K-means Intra-Cluster Variance (Y): {intra_cluster_variance_k_means_for_Y}, K-medoids Intra-Cluster Variance (Y): {intra_cluster_variance_k_medoids_for_Y}")
    intra_cluster_variance_k_means_for_A, cluster_variances_k_means_for_A = intra_cluster_variance_matrices(A, final_k_means_clusters_for_A, centroids_for_A)
    intra_cluster_variance_k_medoids_for_A, cluster_variances_k_medoids_for_A = intra_cluster_variance_matrices(A, final_k_medoids_clusters_for_A, medoids_for_A)
    print(f"K-means Intra-Cluster Variance (Matrices A): {intra_cluster_variance_k_means_for_A}, K-medoids Intra-Cluster Variance (Matrices A): {intra_cluster_variance_k_medoids_for_A}")
    intra_cluster_variance_k_means_for_B, cluster_variances_k_means_for_B = intra_cluster_variance_matrices(B, final_k_means_clusters_for_B, centroids_for_B)
    intra_cluster_variance_k_medoids_for_B, cluster_variances_k_medoids_for_B = intra_cluster_variance_matrices(B, final_k_medoids_clusters_for_B, medoids_for_B)
    print(f"K-means Intra-Cluster Variance (Matrices B): {intra_cluster_variance_k_means_for_B}, K-medoids Intra-Cluster Variance (Matrices B): {intra_cluster_variance_k_medoids_for_B}")

main()



