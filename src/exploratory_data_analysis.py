import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

def plot_distributions(df, columns=None, figsize=(15, 10)):
    """
    Plot distributions of features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        List of columns to plot
    figsize : tuple, optional
        Figure size
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=['number']).columns.tolist()
        # Exclude customer_id if it exists
        if 'customer_id' in columns:
            columns.remove('customer_id')
    
    # Create figure with subplots
    n_cols = 2
    n_rows = (len(columns) + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, column in enumerate(columns):
        # Histogram with KDE
        sns.histplot(df[column], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
    
    # Hide any empty subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('../output/feature_distributions.png', dpi=300)
    plt.show()


def plot_boxplots(df, columns=None, figsize=(15, 10)):
    """
    Plot boxplots to visualize distributions and potential outliers
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        List of columns to plot
    figsize : tuple, optional
        Figure size
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=['number']).columns.tolist()
        # Exclude customer_id if it exists
        if 'customer_id' in columns:
            columns.remove('customer_id')
    
    # Create figure with subplots
    n_cols = 2
    n_rows = (len(columns) + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, column in enumerate(columns):
        # Boxplot
        sns.boxplot(y=df[column], ax=axes[i])
        axes[i].set_title(f'Boxplot of {column}')
        axes[i].set_xlabel(column)
    
    # Hide any empty subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('../output/boxplots.png', dpi=300)
    plt.show()


def plot_correlation_heatmap(df, columns=None, figsize=(10, 8)):
    """
    Plot correlation heatmap
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        List of columns to include
    figsize : tuple, optional
        Figure size
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=['number']).columns.tolist()
        # Exclude customer_id if it exists
        if 'customer_id' in columns:
            columns.remove('customer_id')
    
    # Compute correlation matrix
    corr_matrix = df[columns].corr()
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('../output/correlation_heatmap.png', dpi=300)
    plt.show()
    
    # Return high correlations
    high_corr = []
    for i, row in enumerate(corr_matrix.values):
        for j, corr in enumerate(row):
            if i < j and abs(corr) > 0.5:  # Only upper triangle and significant correlations
                high_corr.append((corr_matrix.index[i], corr_matrix.columns[j], corr))
    
    if high_corr:
        print("\nFeatures with high correlation (|r| > 0.5):")
        for feature1, feature2, corr in high_corr:
            print(f"{feature1} and {feature2}: {corr:.3f}")


def plot_pairplot(df, columns=None, figsize=(15, 15)):
    """
    Create a pairplot to visualize relationships between features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        List of columns to include
    figsize : tuple, optional
        Figure size
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=['number']).columns.tolist()
        # Exclude customer_id if it exists
        if 'customer_id' in columns:
            columns.remove('customer_id')
        
        # If there are too many columns, select only the most relevant ones
        if len(columns) > 5:
            print("Too many columns for pairplot. Selecting the first 5.")
            columns = columns[:5]
    
    plt.figure(figsize=figsize)
    pair_plot = sns.pairplot(df[columns], diag_kind='kde')
    pair_plot.fig.suptitle("Pairwise Relationships", y=1.02)
    plt.tight_layout()
    plt.savefig('../output/pairplot.png', dpi=300)
    plt.show()


def plot_pca_explained_variance(df, columns=None):
    """
    Plot PCA explained variance to determine optimal number of components
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        List of columns to include in PCA
    
    Returns:
    --------
    None
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=['number']).columns.tolist()
        # Exclude customer_id if it exists
        if 'customer_id' in columns:
            columns.remove('customer_id')
    
    # Initialize PCA
    pca = PCA()
    
    # Fit PCA
    pca.fit(df[columns])
    
    # Compute cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    
    # Individual explained variance
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_, 
            alpha=0.7, 
            label='Individual explained variance')
    
    # Cumulative explained variance
    plt.step(range(1, len(cumulative_variance) + 1), 
             cumulative_variance, 
             where='mid', 
             label='Cumulative explained variance')
    
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.grid(linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../output/pca_explained_variance.png', dpi=300)
    plt.show()
    
    # Print explained variance ratio
    print("Explained variance ratio by component:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {ratio:.4f} ({cumulative_variance[i]:.4f} cumulative)")


def plot_pca_visualization(df, columns=None, n_components=2):
    """
    Create PCA visualization of the data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        List of columns to include in PCA
    n_components : int, optional
        Number of PCA components to use (default: 2)
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with PCA components
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=['number']).columns.tolist()
        # Exclude customer_id if it exists
        if 'customer_id' in columns:
            columns.remove('customer_id')
    
    # Initialize and fit PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df[columns])
    
    # Create a dataframe with principal components
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Add customer_id if it exists
    if 'customer_id' in df.columns:
        pca_df['customer_id'] = df['customer_id'].values
    
    # Create 2D scatter plot if we have at least 2 components
    if n_components >= 2:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', data=pca_df)
        plt.title('PCA: First Two Principal Components')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.grid(linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('../output/pca_visualization.png', dpi=300)
        plt.show()
        
        # Create an interactive 3D scatter plot if we have at least 3 components
        if n_components >= 3:
            # Create an interactive 3D scatter plot
            fig = px.scatter_3d(
                pca_df, 
                x='PC1', 
                y='PC2', 
                z='PC3',
                title='PCA: First Three Principal Components',
                labels={
                    'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                    'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
                    'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)'
                }
            )
            fig.update_traces(marker=dict(size=5))
            fig.update_layout(
                scene=dict(
                    xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                    yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                    zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%})'
                )
            )
            fig.write_html('../output/pca_3d_visualization.html')
            fig.show()
    
    # Display feature loadings (coefficients)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_df = pd.DataFrame(
        loadings, 
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=columns
    )
    
    print("\nPCA Feature Loadings:")
    print(loading_df)
    
    # Plot heatmap of feature loadings
    plt.figure(figsize=(10, 8))
    sns.heatmap(loading_df, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('PCA Feature Loadings')
    plt.tight_layout()
    plt.savefig('../output/pca_loadings.png', dpi=300)
    plt.show()
    
    return pca_df


def run_exploratory_analysis(df):
    """
    Run a complete exploratory analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    None
    """
    print("Starting exploratory data analysis...")
    
    # Basic dataset info
    print("\n===== Dataset Overview =====")
    print(f"Shape: {df.shape}")
    print("\nDataset head:")
    print(df.head())
    
    print("\nDataset info:")
    print(df.info())
    
    print("\nDescriptive statistics:")
    print(df.describe().T)
    
    # Check the data types
    print("\nData types:")
    print(df.dtypes)
    
    # Plot distributions
    print("\n===== Plotting Feature Distributions =====")
    plot_distributions(df)
    
    # Plot boxplots
    print("\n===== Plotting Boxplots =====")
    plot_boxplots(df)
    
    # Correlation analysis
    print("\n===== Correlation Analysis =====")
    plot_correlation_heatmap(df)
    
    # Pairplot
    print("\n===== Pairwise Relationships =====")
    plot_pairplot(df)
    
    # PCA analysis
    print("\n===== PCA Analysis =====")
    plot_pca_explained_variance(df)
    pca_df = plot_pca_visualization(df, n_components=3)
    
    print("\nExploratory data analysis completed!")
    
    return pca_df


if __name__ == "__main__":
    from data_preprocessing import preprocess_data
    
    # Example usage
    file_path = "../data/customer_behavior_analytcis.csv"
    
    # Preprocess the data
    original_df, preprocessed_df, scaler = preprocess_data(file_path)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs('../output', exist_ok=True)
    
    # Run exploratory analysis on preprocessed data
    # Exclude customer_id from analysis
    if 'customer_id' in preprocessed_df.columns:
        analysis_df = preprocessed_df.drop(columns=['customer_id'])
    else:
        analysis_df = preprocessed_df
        
    pca_df = run_exploratory_analysis(analysis_df)
    print("EDA completed successfully!")