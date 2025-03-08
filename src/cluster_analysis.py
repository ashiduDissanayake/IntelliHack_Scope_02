import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


def interpret_clusters(original_df, cluster_labels, feature_names):
    """
    Interpret cluster characteristics and identify customer segments
    
    Parameters:
    -----------
    original_df : pd.DataFrame
        Original dataframe with unscaled features
    cluster_labels : array-like
        Cluster labels
    feature_names : list
        List of feature names used for clustering
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with cluster profiles and interpretations
    """
    print("Interpreting clusters...")
    
    # Create a dataframe with original features and cluster labels
    df_with_labels = original_df.copy()
    df_with_labels['Cluster'] = cluster_labels
    
    # Calculate cluster profiles (mean of each feature for each cluster)
    cluster_profiles = df_with_labels.groupby('Cluster')[feature_names].mean()
    
    print("\nCluster Profiles (Mean Values):")
    print(cluster_profiles)
    
    # Calculate relative feature importance for each cluster (compared to overall mean)
    overall_mean = original_df[feature_names].mean()
    relative_importance = cluster_profiles.copy()
    
    for feature in feature_names:
        if overall_mean[feature] != 0:  # Avoid division by zero
            relative_importance[feature] = (cluster_profiles[feature] - overall_mean[feature]) / overall_mean[feature]
        else:
            relative_importance[feature] = cluster_profiles[feature]
    
    print("\nRelative Feature Importance (% difference from overall mean):")
    print(relative_importance.applymap(lambda x: f"{x*100:.2f}%"))
    
    # Plot heatmap of relative feature importance
    plt.figure(figsize=(12, 8))
    sns.heatmap(relative_importance, annot=True, cmap='RdYlGn', center=0, fmt='.2f',
               cbar_kws={'label': 'Relative Importance (% difference from mean)'})
    plt.title('Relative Feature Importance by Cluster')
    plt.tight_layout()
    plt.savefig('../output/cluster_relative_importance.png', dpi=300)
    plt.show()
    
    # Create detailed cluster descriptions based on feature values
    cluster_descriptions = {}
    expected_segments = ['Bargain Hunters', 'High Spenders', 'Window Shoppers']
    segment_match = {}
    
    for cluster_id in cluster_profiles.index:
        profile = cluster_profiles.loc[cluster_id]
        relative = relative_importance.loc[cluster_id]
        
        description = []
        
        # Check if total_purchases is high, moderate, or low
        if profile['total_purchases'] > overall_mean['total_purchases'] * 1.2:
            description.append("High number of purchases")
            purchases_level = "high"
        elif profile['total_purchases'] < overall_mean['total_purchases'] * 0.8:
            description.append("Low number of purchases")
            purchases_level = "low"
        else:
            description.append("Moderate number of purchases")
            purchases_level = "moderate"
            
        # Check avg_cart_value
        if profile['avg_cart_value'] > overall_mean['avg_cart_value'] * 1.2:
            description.append("High average cart value")
            cart_value_level = "high"
        elif profile['avg_cart_value'] < overall_mean['avg_cart_value'] * 0.8:
            description.append("Low average cart value")
            cart_value_level = "low"
        else:
            description.append("Moderate average cart value")
            cart_value_level = "moderate"
            
        # Check total_time_spent
        if profile['total_time_spent'] > overall_mean['total_time_spent'] * 1.2:
            description.append("Spends significant time on platform")
            time_spent_level = "high"
        elif profile['total_time_spent'] < overall_mean['total_time_spent'] * 0.8:
            description.append("Spends less time on platform")
            time_spent_level = "low"
        else:
            description.append("Spends moderate time on platform")
            time_spent_level = "moderate"
            
        # Check product_click
        if profile['product_click'] > overall_mean['product_click'] * 1.2:
            description.append("Views many products")
            product_click_level = "high"
        elif profile['product_click'] < overall_mean['product_click'] * 0.8:
            description.append("Views fewer products")
            product_click_level = "low"
        else:
            description.append("Views a moderate number of products")
            product_click_level = "moderate"
            
        # Check discount_counts
        if 'discount_counts' in profile:
            discount_field = 'discount_counts'
        else:
            discount_field = 'discount_count'
            
        if profile[discount_field] > overall_mean[discount_field] * 1.2:
            description.append("Frequently uses discount codes")
            discount_level = "high"
        elif profile[discount_field] < overall_mean[discount_field] * 0.8:
            description.append("Rarely uses discount codes")
            discount_level = "low"
        else:
            description.append("Sometimes uses discount codes")
            discount_level = "moderate"
        
        # Determine which expected segment this cluster likely corresponds to
        # Bargain Hunters: high purchases, low cart value, moderate time, moderate clicks, high discounts
        # High Spenders: moderate purchases, high cart value, moderate time, moderate clicks, low discounts
        # Window Shoppers: low purchases, moderate cart value, high time, high clicks, low discounts
        
        if (purchases_level == "high" and 
            cart_value_level == "low" and 
            discount_level == "high"):
            segment_match[cluster_id] = "Bargain Hunters"
            
        elif (purchases_level == "moderate" and 
              cart_value_level == "high" and 
              discount_level == "low"):
            segment_match[cluster_id] = "High Spenders"
            
        elif (purchases_level == "low" and 
              time_spent_level == "high" and 
              product_click_level == "high" and
              discount_level == "low"):
            segment_match[cluster_id] = "Window Shoppers"
        
        cluster_descriptions[cluster_id] = {
            'description': description,
            'key_features': {
                'purchases': purchases_level,
                'cart_value': cart_value_level,
                'time_spent': time_spent_level,
                'product_clicks': product_click_level,
                'discounts': discount_level
            }
        }
    
    # Print cluster descriptions
    print("\nCluster Descriptions:")
    for cluster_id, data in cluster_descriptions.items():
        segment_name = segment_match.get(cluster_id, "Unidentified Segment")
        print(f"\nCluster {cluster_id} - {segment_name}")
        print("Characteristics:")
        for desc in data['description']:
            print(f"- {desc}")
        print("Key Feature Levels:", data['key_features'])
    
    # Create a summary dataframe
    summary_data = []
    for cluster_id, data in cluster_descriptions.items():
        segment_name = segment_match.get(cluster_id, "Unidentified Segment")
        row = {
            'Cluster': cluster_id,
            'Segment': segment_name,
            'Characteristics': "; ".join(data['description'])
        }
        for feature, value in data['key_features'].items():
            row[f'{feature.capitalize()} Level'] = value.capitalize()
        summary_data.append(row)
        
    summary_df = pd.DataFrame(summary_data)
    
    return {
        'cluster_profiles': cluster_profiles,
        'relative_importance': relative_importance,
        'cluster_descriptions': cluster_descriptions,
        'segment_match': segment_match,
        'summary': summary_df
    }


def visualize_segment_comparison(cluster_profiles, segment_match, feature_names):
    """
    Create radar charts to compare customer segments
    
    Parameters:
    -----------
    cluster_profiles : pd.DataFrame
        Dataframe with cluster profiles (mean feature values)
    segment_match : dict
        Dictionary mapping cluster IDs to segment names
    feature_names : list
        List of feature names
        
    Returns:
    --------
    None
    """
    # Normalize the data for radar chart
    min_values = cluster_profiles.min()
    max_values = cluster_profiles.max()
    
    radar_df = cluster_profiles.copy()
    for feature in feature_names:
        if max_values[feature] != min_values[feature]:  # Avoid division by zero
            radar_df[feature] = (cluster_profiles[feature] - min_values[feature]) / (max_values[feature] - min_values[feature])
        else:
            radar_df[feature] = 0.5  # Set to middle if all values are the same
    
    # Create radar chart for each segment
    fig = go.Figure()
    
    for cluster_id in radar_df.index:
        segment_name = segment_match.get(cluster_id, f"Cluster {cluster_id}")
        values = radar_df.loc[cluster_id].values.tolist()
        # Close the loop by repeating the first value
        values.append(values[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=feature_names + [feature_names[0]],  # Close the loop
            fill='toself',
            name=segment_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Normalized data range
            )
        ),
        title='Customer Segment Comparison (Normalized Values)',
        showlegend=True
    )
    
    # Save and show
    fig.write_html('../output/segment_comparison.html')
    fig.show()


def create_segment_summary(analysis_results, original_df, cluster_labels):
    """
    Create a comprehensive summary of customer segments
    
    Parameters:
    -----------
    analysis_results : dict
        Results from interpret_clusters function
    original_df : pd.DataFrame
        Original dataframe with unscaled features
    cluster_labels : array-like
        Cluster labels
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with segment summary
    """
    segment_match = analysis_results['segment_match']
    cluster_profiles = analysis_results['cluster_profiles']
    
    # Add cluster and segment labels to the original dataframe
    df_with_segments = original_df.copy()
    df_with_segments['Cluster'] = cluster_labels
    df_with_segments['Segment'] = df_with_segments['Cluster'].map(lambda x: segment_match.get(x, f"Cluster {x}"))
    
    # Calculate additional metrics for each segment
    segment_summary = df_with_segments.groupby('Segment').agg({
        'total_purchases': ['mean', 'median', 'min', 'max', 'count'],
        'avg_cart_value': ['mean', 'median', 'min', 'max'],
        'total_time_spent': ['mean', 'median', 'min', 'max'],
        'product_click': ['mean', 'median', 'min', 'max'],
        'discount_counts' if 'discount_counts' in df_with_segments else 'discount_count': ['mean', 'median', 'min', 'max']
    })
    
    # Calculate revenue potential
    df_with_segments['estimated_revenue'] = df_with_segments['total_purchases'] * df_with_segments['avg_cart_value']
    
    revenue_by_segment = df_with_segments.groupby('Segment').agg({
        'estimated_revenue': ['sum', 'mean', 'median']
    })
    
    # Customer lifetime value could be estimated (simplified version)
    df_with_segments['simplified_clv'] = df_with_segments['estimated_revenue'] * 3  # Assuming 3x multiplier for lifetime
    
    clv_by_segment = df_with_segments.groupby('Segment').agg({
        'simplified_clv': ['mean', 'median', 'sum']
    })
    
    # Create a summary dataframe
    summary_data = []
    for segment in segment_match.values():
        if segment in segment_summary.index:
            row = {
                'Segment': segment,
                'Customer Count': segment_summary.loc[segment, ('total_purchases', 'count')],
                'Avg Purchases': segment_summary.loc[segment, ('total_purchases', 'mean')],
                'Avg Cart Value': segment_summary.loc[segment, ('avg_cart_value', 'mean')],
                'Avg Time Spent': segment_summary.loc[segment, ('total_time_spent', 'mean')],
                'Avg Product Clicks': segment_summary.loc[segment, ('product_click', 'mean')],
                'Avg Discount Usage': segment_summary.loc[segment, 
                    ('discount_counts' if 'discount_counts' in df_with_segments else 'discount_count', 'mean')],
                'Total Revenue Contribution': revenue_by_segment.loc[segment, ('estimated_revenue', 'sum')],
                'Avg Revenue Per Customer': revenue_by_segment.loc[segment, ('estimated_revenue', 'mean')],
                'Avg Est. CLV': clv_by_segment.loc[segment, ('simplified_clv', 'mean')]
            }
            summary_data.append(row)
    
    segment_summary_df = pd.DataFrame(summary_data)
    
    # Calculate percentage contributions
    total_customers = segment_summary_df['Customer Count'].sum()
    total_revenue = segment_summary_df['Total Revenue Contribution'].sum()
    
    segment_summary_df['Customer %'] = (segment_summary_df['Customer Count'] / total_customers * 100).round(1)
    segment_summary_df['Revenue %'] = (segment_summary_df['Total Revenue Contribution'] / total_revenue * 100).round(1)
    
    # Reorder columns
    column_order = ['Segment', 'Customer Count', 'Customer %', 'Total Revenue Contribution', 'Revenue %', 
                   'Avg Purchases', 'Avg Cart Value', 'Avg Time Spent', 'Avg Product Clicks', 
                   'Avg Discount Usage', 'Avg Revenue Per Customer', 'Avg Est. CLV']
    
    segment_summary_df = segment_summary_df[column_order]
    
    # Print segment summary
    print("\nCustomer Segment Summary:")
    print(segment_summary_df.to_string(index=False))
    
    # Create visualization of segment sizes and revenue contribution
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # Segment sizes (customer count)
    ax[0].pie(segment_summary_df['Customer Count'], labels=segment_summary_df['Segment'],
             autopct='%1.1f%%', startangle=90, explode=[0.05] * len(segment_summary_df),
             shadow=True)
    ax[0].set_title('Customer Distribution by Segment')
    
    # Revenue contribution
    ax[1].pie(segment_summary_df['Total Revenue Contribution'], labels=segment_summary_df['Segment'],
             autopct='%1.1f%%', startangle=90, explode=[0.05] * len(segment_summary_df),
             shadow=True)
    ax[1].set_title('Revenue Contribution by Segment')
    
    plt.tight_layout()
    plt.savefig('../output/segment_distribution_and_revenue.png', dpi=300)
    plt.show()
    
    return segment_summary_df


def analyze_clusters(original_df, cluster_labels, feature_names):
    """
    Comprehensive cluster analysis
    
    Parameters:
    -----------
    original_df : pd.DataFrame
        Original dataframe with unscaled features
    cluster_labels : array-like
        Cluster labels
    feature_names : list
        List of feature names
        
    Returns:
    --------
    dict
        Dictionary with analysis results
    """
    # Interpret clusters
    analysis_results = interpret_clusters(original_df, cluster_labels, feature_names)
    
    # Visualize segment comparison
    print("\nVisualizing segment comparison...")
    visualize_segment_comparison(
        analysis_results['cluster_profiles'], 
        analysis_results['segment_match'], 
        feature_names
    )
    
    # Create segment summary
    print("\nCreating segment summary...")
    segment_summary = create_segment_summary(analysis_results, original_df, cluster_labels)
    
    return {
        'analysis_results': analysis_results,
        'segment_summary': segment_summary
    }


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
    
    # Determine features for clustering
    feature_names = [col for col in original_df.columns if col != 'customer_id']
    
    # Exclude customer_id from clustering
    if 'customer_id' in preprocessed_df.columns:
        cluster_df = preprocessed_df.drop(columns=['customer_id'])
    else:
        cluster_df = preprocessed_df
    
    # Run KMeans clustering with 3 clusters
    kmeans_model = kmeans_clustering(cluster_df, n_clusters=3)
    cluster_labels = kmeans_model.labels_
    
    # Analyze clusters
    results = analyze_clusters(original_df, cluster_labels, feature_names)
    
    print("\nCluster analysis completed successfully!")