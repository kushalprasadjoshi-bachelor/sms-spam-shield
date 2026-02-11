"""
Main script to execute the complete data processing pipeline.
This script loads the raw SMS dataset, processes it, clusters spam messages,
balances the dataset, and saves the processed results.
"""
import sys
import os
import warnings
import yaml
import pickle
from pathlib import Path
import pandas as pd

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Suppress warnings
warnings.filterwarnings('ignore')

from ml.dataset_processing.pipeline import DataProcessingPipeline

def load_configs():
    """Load all configuration files."""
    config_dir = project_root / "config" / "dataset_processing"
    
    with open(config_dir / "data_processing.yaml", 'r') as f:
        data_config = yaml.safe_load(f)
    
    with open(config_dir / "clustering.yaml", 'r') as f:
        cluster_config = yaml.safe_load(f)
    
    with open(config_dir / "balancing.yaml", 'r') as f:
        balance_config = yaml.safe_load(f)
    
    with open(config_dir / "paths.yaml", 'r') as f:
        path_config = yaml.safe_load(f)
    
    return {
        'data': data_config,
        'clustering': cluster_config,
        'balancing': balance_config,
        'paths': path_config
    }

def main():
    """Execute the complete data processing pipeline."""
    print("=" * 60)
    print("SMS DATASET PROCESSING PIPELINE")
    print("=" * 60)
    
    # Load configurations
    print("\n📋 Loading configurations...")
    configs = load_configs()
    
    # Create output directories
    output_dirs = [
        project_root / "dataset" / "processed",
        project_root / "reports",
        project_root / "models" / "dataset_processing"
    ]
    
    for dir_path in output_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize and run pipeline
    print("\n🚀 Initializing data processing pipeline...")
    pipeline = DataProcessingPipeline(configs)
    
    # Step 1: Load and preprocess data
    print("\n📊 Step 1: Loading and preprocessing raw data...")
    pipeline.load_data()
    
    # Step 2: Extract legitimate messages
    print("\n🔍 Step 2: Extracting legitimate messages...")
    pipeline.extract_legitimate_messages()
    
    # Step 3: Cluster spam messages
    print("\n🎯 Step 3: Clustering spam messages into subtypes...")
    pipeline.cluster_spam_messages()
    
    # Step 4: Merge and label all messages
    print("\n🔗 Step 4: Merging and labeling all messages...")
    pipeline.merge_and_label()
    
    # Step 5: Balance the dataset
    print("\n⚖️  Step 5: Balancing dataset using SMOTE...")
    pipeline.balance_dataset()
    
    # Step 6: Save processed data
    print("\n💾 Step 6: Saving processed datasets...")
    pipeline.save_processed_data()
    
    # Step 7: Generate reports
    print("\n📈 Step 7: Generating analysis reports...")
    reports = pipeline.generate_reports()
    
    # Save reports
    reports_dir = project_root / "reports"
    for report_name, report_data in reports.items():
        report_path = reports_dir / f"{report_name}.txt"
        with open(report_path, 'w', encoding="utf-8") as f:
            f.write(report_data)
        print(f"  ✓ Saved {report_name} to {report_path}")
    
    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 60)
    
    # Print summary
    print("\n📊 DATASET SUMMARY:")
    print(f"   Total messages processed: {len(pipeline.processed_data)}")
    print(f"   Legitimate messages: {len(pipeline.legitimate_data)}")
    print(f"   Spam messages: {len(pipeline.spam_data)}")
    print(f"   Categories identified: {len(pipeline.processed_data['category'].unique())}")
    print(f"   Categories: {', '.join(sorted(pipeline.processed_data['category'].unique()))}")
    
    # Show category distribution
    print("\n📈 CATEGORY DISTRIBUTION:")
    category_dist = pipeline.processed_data['category'].value_counts()
    for category, count in category_dist.items():
        percentage = (count / len(pipeline.processed_data)) * 100
        print(f"   {category}: {count} messages ({percentage:.1f}%)")
    
    print("\n📁 OUTPUT FILES:")
    print(f"   Original dataset: {configs['paths']['paths']['raw_data']}")
    print(f"   Processed dataset: {configs['paths']['paths']['processed_data']}")
    print(f"   Balanced dataset: {configs['paths']['paths']['processed_data_balanced']}")
    print(f"   Vectorizer: {configs['paths']['paths']['vectorizer_path']}")
    print(f"   Clusterer: {configs['paths']['paths']['clusterer_path']}")
    print("\n🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()