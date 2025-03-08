import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import preprocess_data
from src.exploratory_data_analysis import run_exploratory_analysis
from src.model_building import find_optimal_k, build_clustering_models
from src.model_evaluation import evaluate_clustering_results
from src.cluster_analysis import analyze_clusters
from src.utils import create_output_directory, save_results, export_clusters_to_csv

def main():
    """
    Main function to run the customer segmentation pipeline
    """
    print("Starting Customer Segmentation Analysis...")
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Results will be saved to: {output_dir}")
    
    # 1. Load and preprocess data
    print("\n===== Data Preprocessing =====")
    file_path = "./data/customer_behavior_analytcis.csv"
    original_df, preprocessed_df, scaler = preprocess_data(file_path)
    
    # Save preprocessed data
    preprocessed_df.to_csv(f"{output_dir}/preprocessed_data.csv", index=False)
    
    # 2. Exploratory Data Analysis
    print("\n===== Exploratory Data Analysis =====")
    features = [col for col in preprocessed_df.columns if col != 'customer_id']
    if 'customer_id' in preprocessed_df.columns:
        analysis_df = preprocessed_df.drop(columns=['customer_id'])
    else:
        analysis_df = preprocessed_df
        
    pca_df = run_exploratory_analysis(analysis_df)
    
    # 3. Find optimal number of clusters
    print("\n===== Finding Optimal Number of Clusters =====")
    optimal_k_results = find_optimal_k(analysis_df, max_clusters=10)
    
    # 4. Build and evaluate clustering models
    print("\n===== Building Clustering Models =====")
    # Based on domain knowledge, we know there are 3 segments
    clustering_models = build_clustering_models(analysis_df, n_clusters=3)
    
    # Select the best model (in this case, we'll use KMeans)
    best_model = clustering_models['kmeans']
    cluster_labels = best_model['labels']
    
    # 5. Evaluate clustering results
    print("\n===== Evaluating Clustering Results =====")
    evaluation_results = evaluate_clustering_results(
        analysis_df,
        original_df,
        cluster_labels,
        features
    )
    
    # 6. Analyze clusters and identify customer segments
    print("\n===== Analyzing Customer Segments =====")
    analysis_results = analyze_clusters(original_df, cluster_labels, features)
    
    segment_mapping = analysis_results['analysis_results']['segment_match']
    segment_summary = analysis_results['segment_summary']
    
    # Print the final segment summary
    print("\n===== Customer Segment Summary =====")
    print(segment_summary.to_string())
    
    # 7. Export results
    print("\n===== Exporting Results =====")
    
    # Save segment summary
    segment_summary.to_csv(f"{output_dir}/segment_summary.csv", index=False)
    
    # Export original data with cluster and segment labels
    export_clusters_to_csv(
        original_df,
        cluster_labels,
        segment_mapping,
        file_path=f"{output_dir}/customer_segments.csv"
    )
    
    # Save cluster profiles
    cluster_profiles = evaluation_results['original_cluster_profiles']
    cluster_profiles.to_csv(f"{output_dir}/cluster_profiles.csv")
    
    print("\nCustomer Segmentation Analysis completed successfully!")
    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()