import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import plotly.express as px
import plotly.graph_objects as go

def find_optimal_k(df, max_clusters=10, random_state=42):
    """
    Find optimal number of clusters using multiple methods
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features
    max_clusters : int, optional
        Maximum number of clusters to try
    random_state : int, optional
        Random state for reproducibility
        
    Returns:
    --------
    dict
        Dictionary with evaluation results
    """
    # Initialize empty lists to store results
    inertia = []
    silhouette = []
    calinski_harabasz = []
    davies_bouldin = []
    
    # Range of clusters to try
    K = range(2, max_clusters+1)
    
    for k in K:
        # KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(df)
        
        # Get cluster labels
        labels = kmeans.labels_
        
        # Inertia (within-cluster sum-of-squares)
        inertia.append(kmeans.inertia_)
        
        # Silhouette score
        silhouette.append(silhouette_score(df, labels))
        
        # Calinski-Harabasz Index
        calinski_harabasz.append(calinski_harabasz_score(df, labels))
        
        # Davies-Bouldin Index
        davies_bouldin.append(davies_bouldin_score(df, labels))
    
    # Plot elbow curve
    plt.figure(figsize=(12, 10))
    
    # Inertia (Elbow Method)
    plt.subplot(2, 2, 1)
    plt.plot(K, inertia, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid(True)
    
    # Silhouette Score (higher is better)
    plt.subplot(2, 2, 2)
    plt.plot(K, silhouette, 'go-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score (higher is better)')
    plt.grid(True)
    
    # Calinski-Harabasz Index (higher is better)
    plt.subplot(2, 2, 3)
    plt.plot(K, calinski_harabasz, 'ro-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Calinski-Harabasz Index')
    plt.title('Calinski-Harabasz Index (higher is better)')
    plt.grid(True)
    
    # Davies-Bouldin Index (lower is better)
    plt.subplot(2, 2, 4)
    plt.plot(K, davies_bouldin, 'mo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Davies-Bouldin Index')
    plt.title('Davies-Bouldin Index (lower is better)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../output/optimal_clusters.png', dpi=300)
    plt.show()
    
    # Use yellowbrick for elbow visualization
    plt.figure(figsize=(10, 6))
    visualizer = KElbowVisualizer(KMeans(random_state=random_state), k=(2, max_clusters), timings=False)
    visualizer.fit(df)
    visualizer.finalize()
    plt.savefig('../output/elbow_visualizer.png', dpi=300)
    plt.show()
    
    # Use yellowbrick for silhouette visualization
    for k in [2, 3, 4]:
        plt.figure(figsize=(10, 6))
        model = KMeans(n_clusters=k, random_state=random_state)
        visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
        visualizer.fit(df)
        visualizer.finalize()
        plt.title(f'Silhouette Plot for KMeans with {k} clusters')
        plt.savefig(f'../output/silhouette_k{k}.png', dpi=300)
        plt.show()
    
    # Return the results
    results = {
        'k_values': list(K),
        'inertia': inertia,
        'silhouette': silhouette,
        'calinski_harabasz': calinski_harabasz,
        'davies_bouldin': davies_bouldin
    }
    
    # Print results
    print("\nOptimal number of clusters based on various metrics:")
    print(f"Elbow Method: Look at the plot for the elbow point")
    print(f"Silhouette Score (max): k={K[np.argmax(silhouette)]}, score={max(silhouette):.4f}")
    print(f"Calinski-Harabasz Index (max): k={K[np.argmax(calinski_harabasz)]}, score={max(calinski_harabasz):.4f}")
    print(f"Davies-Bouldin Index (min): k={K[np.argmin(davies_bouldin)]}, score={min(davies_bouldin):.4f}")
    
    return results


def kmeans_clustering(df, n_clusters=3, random_state=42):
    """
    Perform KMeans clustering
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features
    n_clusters : int, optional
        Number of clusters
    random_state : int, optional
        Random state for reproducibility
        
    Returns:
    --------
    KMeans
        Fitted KMeans model
    """
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(df)
    
    return kmeans


def gmm_clustering(df, n_clusters=3, random_state=42):
    """
    Perform Gaussian Mixture Model clustering
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features
    n_clusters : int, optional
        Number of clusters
    random_state : int, optional
        Random state for reproducibility
        
    Returns:
    --------
    GaussianMixture
        Fitted GMM model
    """
    # Initialize and fit GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state, n_init=10)
    gmm.fit(df)
    
    return gmm


def hierarchical_clustering(df, n_clusters=3):
    """
    Perform Hierarchical (Agglomerative) Clustering
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features
    n_clusters : int, optional
        Number of clusters
        
    Returns:
    --------
    AgglomerativeClustering
        Fitted Agglomerative Clustering model
    """
    # Initialize and fit Agglomerative Clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical.fit(df)
    
    return hierarchical


def dbscan_clustering(df, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features
    eps : float, optional
        The maximum distance between two samples for them to be considered as in the same neighborhood
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered as a core point
        
    Returns:
    --------
    DBSCAN
        Fitted DBSCAN model
    """
    # Initialize and fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(df)
    
    return dbscan


def visualize_clusters_2d(df, labels, title="Cluster Visualization", save_path=None):
    """
    Visualize clusters in 2D using PCA
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features
    labels : array-like
        Cluster labels
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the visualization
        
    Returns:
    --------
    None
    """
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df)
    
    # Create a dataframe with principal components and cluster labels
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=['PC1', 'PC2']
    )
    pca_df['Cluster'] = labels
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=80, alpha=0.8)
    plt.title(title)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(linestyle='--', alpha=0.5)
    plt.legend(title='Cluster')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()
    
    # Create interactive scatter plot
    fig = px.scatter(
        pca_df, 
        x='PC1', 
        y='PC2', 
        color='Cluster',
        title=title,
        labels={
            'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'
        }
    )
    fig.update_traces(marker=dict(size=10))
    
    if save_path:
        html_path = save_path.replace('.png', '.html')
        fig.write_html(html_path)
    
    fig.show()


def visualize_clusters_3d(df, labels, title="3D Cluster Visualization", save_path=None):
    """
    Visualize clusters in 3D using PCA
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features
    labels : array-like
        Cluster labels
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the visualization
        
    Returns:
    --------
    None
    """
    # Apply PCA for visualization
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(df)
    
    # Create a dataframe with principal components and cluster labels
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=['PC1', 'PC2', 'PC3']
    )
    pca_df['Cluster'] = labels
    
    # Create interactive 3D scatter plot
    fig = px.scatter_3d(
        pca_df, 
        x='PC1', 
        y='PC2', 
        z='PC3',
        color='Cluster',
        title=title,
        labels={
            'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
            'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)'
        }
    )
    fig.update_traces(marker=dict(size=5))
    
    if save_path:
        html_path = save_path.replace('.png', '.html')
        fig.write_html(html_path)
    
    fig.show()


def build_clustering_models(df, n_clusters=3, random_state=42):
    """
    Build and compare multiple clustering models
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features
    n_clusters : int, optional
        Number of clusters
    random_state : int, optional
        Random state for reproducibility
        
    Returns:
    --------
    dict
        Dictionary with fitted models and labels
    """
    print(f"Building clustering models with {n_clusters} clusters...")
    
    # KMeans
    print("\nFitting KMeans...")
    kmeans_model = kmeans_clustering(df, n_clusters=n_clusters, random_state=random_state)
    kmeans_labels = kmeans_model.labels_
    kmeans_centers = kmeans_model.cluster_centers_
    
    # GMM
    print("Fitting Gaussian Mixture Model...")
    gmm_model = gmm_clustering(df, n_clusters=n_clusters, random_state=random_state)
    gmm_labels = gmm_model.predict(df)
    gmm_centers = gmm_model.means_
    
    # Hierarchical Clustering
    print("Fitting Hierarchical Clustering...")
    hierarchical_model = hierarchical_clustering(df, n_clusters=n_clusters)
    hierarchical_labels = hierarchical_model.labels_
    
    # DBSCAN (with default parameters, might need tuning)
    print("Fitting DBSCAN...")
    dbscan_model = dbscan_clustering(df, eps=0.5, min_samples=5)
    dbscan_labels = dbscan_model.labels_
    
    # Number of clusters in DBSCAN (including noise points labeled as -1)
    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"DBSCAN found {n_dbscan_clusters} clusters and {list(dbscan_labels).count(-1)} noise points")
    
    # Calculate silhouette scores
    print("\nCalculating silhouette scores...")
    kmeans_silhouette = silhouette_score(df, kmeans_labels)
    gmm_silhouette = silhouette_score(df, gmm_labels)
    hierarchical_silhouette = silhouette_score(df, hierarchical_labels)
    
    if len(set(dbscan_labels)) > 1 and -1 not in dbscan_labels:
        dbscan_silhouette = silhouette_score(df, dbscan_labels)
    elif len(set(dbscan_labels)) > 1:
        # Calculate silhouette score excluding noise points
        mask = dbscan_labels != -1
        if sum(mask) > 1:  # Ensure we have at least 2 points
            dbscan_silhouette = silhouette_score(df[mask], dbscan_labels[mask])
        else:
            dbscan_silhouette = float('nan')
    else:
        dbscan_silhouette = float('nan')
    
    # Print silhouette scores
    print(f"KMeans Silhouette Score: {kmeans_silhouette:.4f}")
    print(f"GMM Silhouette Score: {gmm_silhouette:.4f}")
    print(f"Hierarchical Silhouette Score: {hierarchical_silhouette:.4f}")
    print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.4f}")
    
    # Visualize clusters
    print("\nVisualizing clusters...")
    visualize_clusters_2d(df, kmeans_labels, title=f"KMeans Clustering (k={n_clusters})", 
                        save_path=f"../output/kmeans_clusters_{n_clusters}.png")
    visualize_clusters_2d(df, gmm_labels, title=f"GMM Clustering (k={n_clusters})", 
                        save_path=f"../output/gmm_clusters_{n_clusters}.png")
    visualize_clusters_2d(df, hierarchical_labels, title=f"Hierarchical Clustering (k={n_clusters})", 
                        save_path=f"../output/hierarchical_clusters_{n_clusters}.png")
    visualize_clusters_2d(df, dbscan_labels, title=f"DBSCAN Clustering (eps=0.5, min_samples=5)", 
                        save_path=f"../output/dbscan_clusters.png")
    
    # 3D visualizations
    visualize_clusters_3d(df, kmeans_labels, title=f"KMeans Clustering 3D (k={n_clusters})", 
                        save_path=f"../output/kmeans_clusters_3d_{n_clusters}.png")
    
    # Return models and labels
    models = {
        'kmeans': {
            'model': kmeans_model,
            'labels': kmeans_labels,
            'centers': kmeans_centers,
            'silhouette': kmeans_silhouette
        },
        'gmm': {
            'model': gmm_model,
            'labels': gmm_labels,
            'centers': gmm_centers,
            'silhouette': gmm_silhouette
        },
        'hierarchical': {
            'model': hierarchical_model,
            'labels': hierarchical_labels,
            'silhouette': hierarchical_silhouette
        },
        'dbscan': {
            'model': dbscan_model,
            'labels': dbscan_labels,
            'silhouette': dbscan_silhouette
        }
    }
    
    return models


if __name__ == "__main__":
    from data_preprocessing import preprocess_data
    
    # Example usage
    file_path = "../data/customer_behavior_analytcis.csv"
    
    # Preprocess the data
    original_df, preprocessed_df, scaler = preprocess_data(file_path)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs('../output', exist_ok=True)
    
    # Exclude customer_id from clustering
    if 'customer_id' in preprocessed_df.columns:
        cluster_df = preprocessed_df.drop(columns=['customer_id'])
    else:
        cluster_df = preprocessed_df
    
    # Find optimal number of clusters
    optimal_k_results = find_optimal_k(cluster_df, max_clusters=10)
    
    # Based on the optimal k analysis and domain knowledge (we know there are 3 segments),
    # let's build the clustering models with 3 clusters
    clustering_models = build_clustering_models(cluster_df, n_clusters=3)
    
    print("Model building completed successfully!")