import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


def create_output_directory():
    """
    Create output directory for saving results
    
    Returns:
    --------
    str
        Path to the output directory
    """
    output_dir = '../output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamped subdirectory for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = f"{output_dir}/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir


def save_results(results, output_dir, prefix=''):
    """
    Save analysis results to files
    
    Parameters:
    -----------
    results : dict
        Dictionary with analysis results
    output_dir : str
        Directory to save results
    prefix : str, optional
        Prefix for filenames
        
    Returns:
    --------
    None
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataframes to CSV
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            filename = f"{prefix}_{key}.csv" if prefix else f"{key}.csv"
            filepath = os.path.join(output_dir, filename)
            value.to_csv(filepath, index=True)
            print(f"Saved {filepath}")
    
    # Save dictionary as JSON (excluding dataframes and non-serializable objects)
    json_results = {}
    for key, value in results.items():
        if isinstance(value, (int, float, str, list, dict, tuple, bool)):
            json_results[key] = value
        elif isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
    
    if json_results:
        filename = f"{prefix}_results.json" if prefix else "results.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
            print(f"Saved {filepath}")


def plot_feature_correlations(df, title='Feature Correlations', figsize=(10, 8), save_path=None):
    """
    Plot correlation matrix heatmap for features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    None
    """
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5)
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()


def plot_3d_scatter(df, x_col, y_col, z_col, color_col=None, title=None, save_path=None):
    """
    Create interactive 3D scatter plot
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    z_col : str
        Column name for z-axis
    color_col : str, optional
        Column name for coloring points
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    None
    """
    if title is None:
        title = f'3D Scatter Plot: {x_col} vs {y_col} vs {z_col}'
    
    fig = px.scatter_3d(
        df, x=x_col, y=y_col, z=z_col, color=color_col,
        title=title, opacity=0.7
    )
    
    fig.update_traces(marker=dict(size=5))
    
    if save_path:
        fig.write_html(save_path)
    
    fig.show()


def create_segment_profile_radar(segment_profiles, feature_names, title='Segment Profiles', save_path=None):
    """
    Create radar chart for segment profiles
    
    Parameters:
    -----------
    segment_profiles : pd.DataFrame
        DataFrame with segment profiles (features as columns, segments as rows)
    feature_names : list
        List of feature names to include
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    None
    """
    # Normalize values for radar chart
    profiles_norm = segment_profiles.copy()
    
    for feature in feature_names:
        min_val = profiles_norm[feature].min()
        max_val = profiles_norm[feature].max()
        if max_val > min_val:
            profiles_norm[feature] = (profiles_norm[feature] - min_val) / (max_val - min_val)
    
    # Create radar chart
    fig = go.Figure()
    
    for idx, segment in enumerate(profiles_norm.index):
        values = profiles_norm.loc[segment, feature_names].tolist()
        # Close the loop
        values.append(values[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=feature_names + [feature_names[0]],
            fill='toself',
            name=f'Segment {segment}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=title,
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
    
    fig.show()


def map_clusters_to_segments(cluster_profiles, expected_segments):
    """
    Map cluster IDs to meaningful segment names
    
    Parameters:
    -----------
    cluster_profiles : pd.DataFrame
        DataFrame with cluster profiles (features as columns, clusters as rows)
    expected_segments : list
        List of expected segment names
        
    Returns:
    --------
    dict
        Mapping of cluster IDs to segment names
    """
    # This is a simplified approach that ranks clusters by their key characteristics
    # and maps them to expected segments based on the problem description
    
    n_clusters = len(cluster_profiles)
    n_segments = len(expected_segments)
    
    if n_clusters != n_segments:
        print(f"Warning: Number of clusters ({n_clusters}) doesn't match number of expected segments ({n_segments})")
    
    # Create feature rankings for each cluster
    rankings = {}
    
    # For each feature, rank clusters from highest to lowest
    for feature in cluster_profiles.columns:
        # Sort clusters by feature value (descending)
        sorted_clusters = cluster_profiles.sort_values(by=feature, ascending=False).index.tolist()
        # Assign ranks (0 = highest)
        ranks = {cluster: rank for rank, cluster in enumerate(sorted_clusters)}
        rankings[feature] = ranks
    
    # Characteristics of expected segments:
    # 1. Bargain Hunters: High purchases, Low avg_cart_value, High discount_count
    # 2. High Spenders: Moderate purchases, High avg_cart_value, Low discount_count
    # 3. Window Shoppers: Low purchases, High time_spent, High product_click, Low purchases
    
    # Calculate scores for each cluster-segment pair
    scores = {}
    
    for cluster in cluster_profiles.index:
        scores[cluster] = {}
        
        # Bargain Hunters score
        bargain_score = 0
        if 'total_purchases' in rankings:
            bargain_score -= rankings['total_purchases'][cluster]  # Higher purchases = better
        if 'avg_cart_value' in rankings:
            bargain_score += rankings['avg_cart_value'][cluster]   # Lower cart value = better
        if 'discount_counts' in rankings:
            bargain_score -= rankings['discount_counts'][cluster]  # Higher discount count = better
        elif 'discount_count' in rankings:
            bargain_score -= rankings['discount_count'][cluster]   # Higher discount count = better
        scores[cluster]['Bargain Hunters'] = bargain_score
        
        # High Spenders score
        spender_score = 0
        if 'avg_cart_value' in rankings:
            spender_score -= rankings['avg_cart_value'][cluster]   # Higher cart value = better
        if 'discount_counts' in rankings:
            spender_score += rankings['discount_counts'][cluster]  # Lower discount count = better
        elif 'discount_count' in rankings:
            spender_score += rankings['discount_count'][cluster]   # Lower discount count = better
        scores[cluster]['High Spenders'] = spender_score
        
        # Window Shoppers score
        shopper_score = 0
        if 'total_purchases' in rankings:
            shopper_score += rankings['total_purchases'][cluster]  # Lower purchases = better
        if 'total_time_spent' in rankings:
            shopper_score -= rankings['total_time_spent'][cluster] # Higher time spent = better
        if 'product_click' in rankings:
            shopper_score -= rankings['product_click'][cluster]    # Higher product clicks = better
        scores[cluster]['Window Shoppers'] = shopper_score
    
    # Assign segments to clusters
    assigned_segments = {}
    assigned_clusters = set()
    
    # Sort segments by their distinctiveness
    segment_distinctiveness = {
        segment: max(scores[cluster][segment] for cluster in scores) - 
                min(scores[cluster][segment] for cluster in scores)
        for segment in expected_segments
    }
    
    # Assign segments in order of distinctiveness
    for segment in sorted(expected_segments, key=lambda s: segment_distinctiveness[s], reverse=True):
        # Find best unassigned cluster for this segment
        best_cluster = None
        best_score = float('-inf')
        
        for cluster in scores:
            if cluster not in assigned_clusters and scores[cluster][segment] > best_score:
                best_cluster = cluster
                best_score = scores[cluster][segment]
        
        if best_cluster is not None:
            assigned_segments[best_cluster] = segment
            assigned_clusters.add(best_cluster)
    
    return assigned_segments


def create_segment_summary_table(summary_df, save_path=None):
    """
    Create a formatted HTML table for segment summary
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Segment summary dataframe
    save_path : str, optional
        Path to save the HTML table
        
    Returns:
    --------
    None
    """
    # Format numeric columns
    formatted_df = summary_df.copy()
    
    # Format currency columns
    currency_cols = ['Avg Cart Value', 'Total Revenue Contribution', 'Avg Revenue Per Customer', 'Avg Est. CLV']
    for col in currency_cols:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].map('${:,.2f}'.format)
    
    # Format percentage columns
    pct_cols = ['Customer %', 'Revenue %']
    for col in pct_cols:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].map('{:.1f}%'.format)
    
    # Format decimal columns
    decimal_cols = ['Avg Purchases', 'Avg Time Spent', 'Avg Product Clicks', 'Avg Discount Usage']
    for col in decimal_cols:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].map('{:.2f}'.format)
    
    # Create HTML table with styling
    html_table = formatted_df.to_html(index=False, classes='segment-table')
    
    # Add CSS styling
    html = f"""
    <style>
    .segment-table {{
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 20px;
        font-family: Arial, sans-serif;
    }}
    .segment-table th {{
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        text-align: left;
        padding: 10px;
        border: 1px solid #ddd;
    }}
    .segment-table td {{
        text-align: left;
        padding: 8px;
        border: 1px solid #ddd;
    }}
    .segment-table tr:nth-child(even) {{
        background-color: #f2f2f2;
    }}
    .segment-table tr:hover {{
        background-color: #ddd;
    }}
    </style>
    <h2>Customer Segment Summary</h2>
    {html_table}
    """
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(html)
            print(f"Saved HTML table to {save_path}")
    
    return html


def export_clusters_to_csv(df, cluster_labels, segment_mapping=None, file_path='../output/customer_segments.csv'):
    """
    Export original data with cluster and segment labels
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe
    cluster_labels : array-like
        Cluster labels
    segment_mapping : dict, optional
        Mapping of cluster IDs to segment names
    file_path : str, optional
        Path to save the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with cluster and segment labels
    """
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Add cluster labels
    result_df['Cluster'] = cluster_labels
    
    # Add segment names if mapping is provided
    if segment_mapping is not None:
        result_df['Segment'] = result_df['Cluster'].map(lambda x: segment_mapping.get(x, f"Cluster {x}"))
    
    # Save to CSV
    result_df.to_csv(file_path, index=False)
    print(f"Exported segmented customers to {file_path}")
    
    return result_df


if __name__ == "__main__":
    print("Utils module loaded successfully!")