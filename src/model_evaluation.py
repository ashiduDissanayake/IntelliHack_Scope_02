import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go


def evaluate_clustering_model(df, cluster_labels):
    """
    Evaluate the clustering model using various metrics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features
    cluster_labels : array-like
        Cluster labels
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Calculate evaluation metrics
    metrics = {
        'silhouette_score': silhouette_score(df, cluster_labels),
        'calinski_harabasz_score': calinski_harabasz_score(df, cluster_labels),
        'davies_bouldin_score': davies_bouldin_score(df, cluster_labels)
    }
    
    # Print evaluation results
    print("\nClustering Evaluation Metrics:")
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f} (higher is better, range: [-1, 1])")
    print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.4f} (higher is better)")
    print(f"Davies-Bouldin Index: {metrics['davies_bouldin_score']:.4f} (lower is better)")
    
    return metrics


def plot_cluster_distributions(df, cluster_labels, features, figsize=(15, 12)):
    """
    Plot the distribution of features across clusters
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features
    cluster_labels : array-like
        Cluster labels
    features : list
        List of feature names to plot
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    None
    """
    # Add cluster labels to the dataframe
    df_with_labels = df.copy()
    df_with_labels['Cluster'] = cluster_labels
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(features), 1, figsize=figsize)
    
    for i, feature in enumerate(features):
        # Create boxplot for the feature across clusters
        sns.boxplot(x='Cluster', y=feature, data=df_with_labels, palette='viridis', ax=axes[i])
        axes[i].set_title(f'Distribution of {feature} across Clusters')
        axes[i].grid(linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'../output/cluster_distributions.png', dpi=300)
    plt.show()


def plot_cluster_profiles(df, cluster_labels, features, figsize=(12, 8)):
    """
    Plot the profile of each cluster
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features
    cluster_labels : array-like
        Cluster labels
    features : list
        List of feature names to plot
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with cluster profiles
    """
    # Add cluster labels to the dataframe
    df_with_labels = df.copy()
    df_with_labels['Cluster'] = cluster_labels
    
    # Get cluster profiles (mean of each feature for each cluster)
    cluster_profiles = df_with_labels.groupby('Cluster')[features].mean()
    
    # Normalize the profiles for radar chart
    scaler = StandardScaler()
    cluster_profiles_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_profiles),
        index=cluster_profiles.index,
        columns=cluster_profiles.columns
    )
    
    # Plot cluster profiles
    plt.figure(figsize=figsize)
    
    # Plot heatmap of cluster profiles
    sns.heatmap(cluster_profiles, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Cluster Profiles (Mean Values)')
    plt.tight_layout()
    plt.savefig(f'../output/cluster_profiles_heatmap.png', dpi=300)
    plt.show()
    
    # Plot radar chart for each cluster
    num_clusters = len(cluster_profiles)
    
    # Create radar chart
    fig = go.Figure()
    
    for cluster_idx in range(num_clusters):
        values = cluster_profiles_scaled.iloc[cluster_idx].values.tolist()
        # Close the loop by repeating the first value
        values.append(values[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=features + [features[0]],  # Close the loop
            fill='toself',
            name=f'Cluster {cluster_idx}',
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-2, 2]  # Standardized data typically falls in this range
            )
        ),
        title='Cluster Profiles (Standardized)',
        showlegend=True
    )
    
    # Save and show
    fig.write_html('../output/cluster_profiles_radar.html')
    fig.show()
    
    # Create bar chart for easier comparison
    plt.figure(figsize=figsize)
    cluster_profiles.T.plot(kind='bar', figsize=figsize)
    plt.title('Cluster Profiles (Mean Feature Values)')
    plt.xlabel('Features')
    plt.ylabel('Mean Value')
    plt.grid(linestyle='--', alpha=0.5)
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(f'../output/cluster_profiles_bar.png', dpi=300)
    plt.show()
    
    return cluster_profiles


def plot_cluster_size(cluster_labels, figsize=(10, 6)):
    """
    Plot the size of each cluster
    
    Parameters:
    -----------
    cluster_labels : array-like
        Cluster labels
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    pd.Series
        Series with cluster sizes
    """
    # Count the number of samples in each cluster
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    
    # Calculate the percentage of samples in each cluster
    total_samples = len(cluster_labels)
    cluster_percentages = (cluster_sizes / total_samples * 100).round(1)
    
    # Create labels for the pie chart
    labels = [f"Cluster {i} ({size} samples, {cluster_percentages[i]}%)" 
              for i, size in cluster_sizes.items()]
    
    # Create pie chart
    plt.figure(figsize=figsize)
    plt.pie(cluster_sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
            shadow=True, explode=[0.05] * len(cluster_sizes))
    plt.title('Cluster Size Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(f'../output/cluster_sizes.png', dpi=300)
    plt.show()
    
    # Create bar chart for cluster sizes
    plt.figure(figsize=figsize)
    plt.bar(range(len(cluster_sizes)), cluster_sizes.values, color='skyblue')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    plt.title('Cluster Size Distribution')
    plt.xticks(range(len(cluster_sizes)), [f'Cluster {i}' for i in cluster_sizes.index])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of the bars
    for i, v in enumerate(cluster_sizes.values):
        plt.text(i, v + 5, str(v), ha='center')
        
    plt.tight_layout()
    plt.savefig(f'../output/cluster_sizes_bar.png', dpi=300)
    plt.show()
    
    return cluster_sizes


def evaluate_clustering_results(df, original_df, cluster_labels, features):
    """
    Comprehensive evaluation of clustering results
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe with features used for clustering
    original_df : pd.DataFrame
        Original dataframe with unscaled features
    cluster_labels : array-like
        Cluster labels
    features : list
        List of feature names used for clustering
        
    Returns:
    --------
    dict
        Dictionary with evaluation results
    """
    print("Evaluating clustering results...")
    
    # Calculate evaluation metrics
    metrics = evaluate_clustering_model(df, cluster_labels)
    
    # Plot cluster distributions
    print("\nPlotting feature distributions across clusters...")
    plot_cluster_distributions(df, cluster_labels, features)
    
    # Plot cluster profiles
    print("\nPlotting cluster profiles...")
    cluster_profiles = plot_cluster_profiles(df, cluster_labels, features)
    
    # Plot cluster sizes
    print("\nPlotting cluster sizes...")
    cluster_sizes = plot_cluster_size(cluster_labels)
    
    # Original (unscaled) feature distributions by cluster
    print("\nPlotting original feature distributions across clusters...")
    original_features = [f for f in features if f in original_df.columns]
    
    # Create a dataframe with original features and cluster labels
    original_with_labels = original_df.loc[df.index].copy()
    original_with_labels['Cluster'] = cluster_labels

    # Plot distribution of original features by cluster
    for feature in original_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Cluster', y=feature, data=original_with_labels, palette='viridis')
        plt.title(f'Distribution of {feature} across Clusters (Original Scale)')
        plt.grid(linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'../output/original_{feature}_distribution.png', dpi=300)
        plt.close()
    
    # Calculate cluster profiles on original scale
    original_cluster_profiles = original_with_labels.groupby('Cluster')[original_features].mean()
    
    print("\nCluster Profiles (Original Scale):")
    print(original_cluster_profiles)
    
    # Combine all evaluation results
    evaluation_results = {
        'metrics': metrics,
        'cluster_profiles': cluster_profiles,
        'original_cluster_profiles': original_cluster_profiles,
        'cluster_sizes': cluster_sizes
    }
    
    return evaluation_results


if __name__ == "__main__":
    from data_preprocessing import preprocess_data
    from model_building import kmeans_clustering
    
    # Example usage
    file_path = "../data/customer_behavior_analytcis.csv"
    
    # Preprocess the data
    original_df, preprocessed_df, scaler = preprocess_data(file_path)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs('../output', exist_ok=True)
    
    # Exclude customer_id from clustering
    features = [col for col in preprocessed_df.columns if col != 'customer_id']
    if 'customer_id' in preprocessed_df.columns:
        cluster_df = preprocessed_df.drop(columns=['customer_id'])
    else:
        cluster_df = preprocessed_df
    
    # Run KMeans clustering with 3 clusters
    kmeans_model = kmeans_clustering(cluster_df, n_clusters=3)
    cluster_labels = kmeans_model.labels_
    
    # Evaluate clustering results
    evaluation_results = evaluate_clustering_results(
        cluster_df, 
        original_df, 
        cluster_labels, 
        features
    )
    
    print("Evaluation completed successfully!")