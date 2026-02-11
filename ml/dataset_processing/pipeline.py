"""
Main pipeline execution module for processing SMS dataset.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from .clusterer import SpamSubtypeClusterer
from .balancer import DatasetBalancer
from .validator import DatasetValidator

class DataProcessingPipeline:
    """Main pipeline for processing SMS dataset."""
    
    def __init__(self, configs: Dict):
        """
        Initialize the data processing pipeline.
        
        Args:
            configs: Dictionary containing all configuration parameters
        """
        self.configs = configs
        self.raw_data = None
        self.legitimate_data = None
        self.spam_data = None
        self.processed_data = None
        self.balanced_data = None
        self.vectorizer = None
        self.clusterer = None
        self.label_encoder = None
        
    def load_data(self) -> None:
        """
        Load raw SMS dataset from CSV file.
        Preserves the original message text exactly as provided.
        """
        raw_path = Path(self.configs['paths']['paths']['raw_data'])
        
        # Read the CSV file
        # Note: The dataset has extra commas, so we need to handle that
        try:
            # Read with proper handling of extra commas
            self.raw_data = pd.read_csv(raw_path, header=None, names=['label', 'message_text'], sep=',', encoding="latin-1")
            
            # Clean the text - remove trailing commas that might have been included
            self.raw_data['message_text'] = self.raw_data['message_text'].apply(
                lambda x: re.sub(r',+$', '', str(x)) if pd.notnull(x) else ''
            )

            # Remove rows with empty text
            self.raw_data = self.raw_data[self.raw_data['message_text'].str.strip() != '']
            
            print(f"  ✓ Loaded {len(self.raw_data)} messages from {raw_path}")
            print(f"  ✓ Columns: {list(self.raw_data.columns)}")
            print(f"  ✓ Labels: {self.raw_data['label'].unique()}")
            
            # Show sample
            print("\n  📝 Sample messages:")
            for i, (label, text) in enumerate(self.raw_data.head(3).values):
                print(f"    {label}: {text[:50]}...")
                
        except Exception as e:
            print(f"  ✗ Error loading data: {e}")
            raise
    
    def extract_legitimate_messages(self) -> None:
        """
        Extract legitimate (ham) messages directly.
        These will be mapped to 'legitimate' category.
        """
        self.legitimate_data = self.raw_data[self.raw_data['label'] == 'ham'].copy()
        self.legitimate_data['category'] = 'legitimate'
        
        # Keep only necessary columns
        self.legitimate_data = self.legitimate_data[['message_text', 'category']]
        
        print(f"  ✓ Extracted {len(self.legitimate_data)} legitimate messages")
        
        # Show distribution
        print(f"  📊 Legitimate messages: {len(self.legitimate_data)}")
        print(f"  📊 Total messages in dataset: {len(self.raw_data)}")
    
    def cluster_spam_messages(self) -> None:
        """
        Cluster spam messages into subtypes using unsupervised learning.
        """
        # Extract spam messages
        self.spam_data = self.raw_data[self.raw_data['label'] == 'spam'].copy()
        
        print(f"  ✓ Extracted {len(self.spam_data)} spam messages for clustering")
        
        if len(self.spam_data) == 0:
            print("  ⚠️  No spam messages found!")
            return
        
        # Initialize clusterer
        cluster_config = self.configs['clustering']['clustering']
        self.clusterer = SpamSubtypeClusterer(cluster_config)
        
        # Perform clustering
        cluster_labels, cluster_centers = self.clusterer.fit_predict(
            self.spam_data['message_text'].tolist()
        )
        
        # Add cluster labels to spam data
        self.spam_data['cluster'] = cluster_labels
        
        # Interpret clusters and assign meaningful labels
        self._interpret_clusters(cluster_centers)
        
        print(f"  ✓ Clustered spam messages into {len(self.spam_data['category'].unique())} categories")
        
        # Show cluster distribution
        print("\n  📊 Spam Cluster Distribution:")
        for category in sorted(self.spam_data['category'].unique()):
            count = len(self.spam_data[self.spam_data['category'] == category])
            print(f"    {category}: {count} messages")
    
    def _interpret_clusters(self, cluster_centers: np.ndarray) -> None:
        """
        Interpret clusters and assign meaningful category labels.
        
        Args:
            cluster_centers: Cluster centers from KMeans
        """
        # Get feature names from vectorizer
        feature_names = self.clusterer.vectorizer.get_feature_names_out()
        
        # Keywords for each category
        category_keywords = {
            'phishing': ['win', 'prize', 'cash', 'claim', 'urgent', 'call', 'winner',
                        'selected', 'award', 'free', 'claim', 'guaranteed', 'won'],
            'promotional': ['free', 'offer', 'discount', 'sale', 'buy', 'shop',
                          'special', 'deal', 'price', 'save', 'limited', 'order'],
            'transactional': ['account', 'bank', 'payment', 'transaction', 'bill',
                            'invoice', 'receipt', 'balance', 'card', 'verify',
                            'security', 'login', 'password'],
            'spam': ['sex', 'dating', 'chat', 'adult', 'sexy', 'hot', 'single',
                    'meet', 'girl', 'guy', 'love', 'romance']
        }
        
        # Assign categories based on top words in each cluster
        for cluster_id in range(len(cluster_centers)):
            # Get top 20 words for this cluster
            top_indices = cluster_centers[cluster_id].argsort()[-20:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            
            # Calculate scores for each category
            category_scores = {}
            for category, keywords in category_keywords.items():
                score = sum(1 for word in top_words if word in keywords)
                category_scores[category] = score
            
            # Assign the category with highest score
            assigned_category = max(category_scores, key=category_scores.get)
            
            # If score is 0, assign as generic spam
            if category_scores[assigned_category] == 0:
                assigned_category = 'spam'
            
            # Update the spam data
            self.spam_data.loc[self.spam_data['cluster'] == cluster_id, 'category'] = assigned_category
        
        # Keep only necessary columns
        self.spam_data = self.spam_data[['message_text', 'category']]
    
    def merge_and_label(self) -> None:
        """
        Merge legitimate and spam messages into a single dataset.
        """
        # Combine both datasets
        self.processed_data = pd.concat([self.legitimate_data, self.spam_data], ignore_index=True)
        
        # Create label encoder for categories
        self.label_encoder = LabelEncoder()
        self.processed_data['label_encoded'] = self.label_encoder.fit_transform(
            self.processed_data['category']
        )
        
        print(f"  ✓ Merged dataset has {len(self.processed_data)} total messages")
        print(f"  ✓ Categories: {', '.join(sorted(self.processed_data['category'].unique()))}")
    
    def balance_dataset(self) -> None:
        """
        Balance the dataset using SMOTE or other balancing techniques.
        """
        balance_config = self.configs['balancing']['balancing']
        balancer = DatasetBalancer(balance_config)
        
        # Prepare data for balancing
        X = self.processed_data['message_text']
        y = self.processed_data['label_encoded']
        
        # Balance the dataset
        X_balanced, y_balanced = balancer.balance(X, y)
        
        # Create balanced dataframe
        self.balanced_data = pd.DataFrame({
            'message_text': X_balanced,
            'category': self.label_encoder.inverse_transform(y_balanced)
        })
        
        print(f"  ✓ Balanced dataset has {len(self.balanced_data)} messages")
        print(f"  ✓ Original class distribution:")
        original_counts = self.processed_data['category'].value_counts()
        for cat, count in original_counts.items():
            print(f"      {cat}: {count}")
        
        print(f"  ✓ Balanced class distribution:")
        balanced_counts = self.balanced_data['category'].value_counts()
        for cat, count in balanced_counts.items():
            print(f"      {cat}: {count}")
    
    def save_processed_data(self) -> None:
        """
        Save processed datasets and trained models.
        """
        paths = self.configs['paths']['paths']
        
        # Save processed data (unbalanced)
        processed_path = Path(paths['processed_data'])
        self.processed_data[['message_text', 'category']].to_csv(processed_path, index=False)
        print(f"  ✓ Saved processed data to {processed_path}")
        
        # Save balanced data
        if self.balanced_data is not None:
            balanced_path = Path(paths['processed_data_balanced'])
            self.balanced_data.to_csv(balanced_path, index=False)
            print(f"  ✓ Saved balanced data to {balanced_path}")
        
        # Save vectorizer
        if self.clusterer and self.clusterer.vectorizer:
            vectorizer_path = Path(paths['vectorizer_path'])
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.clusterer.vectorizer, f)
            print(f"  ✓ Saved TF-IDF vectorizer to {vectorizer_path}")
        
        # Save clusterer
        if self.clusterer:
            clusterer_path = Path(paths['clusterer_path'])
            with open(clusterer_path, 'wb') as f:
                pickle.dump(self.clusterer, f)
            print(f"  ✓ Saved clusterer to {clusterer_path}")
        
        # Save label encoder
        if self.label_encoder:
            encoder_path = Path(paths['label_encoder_path'])
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print(f"  ✓ Saved label encoder to {encoder_path}")
    
    def generate_reports(self) -> Dict[str, str]:
        """
        Generate analysis reports for the processed dataset.
        
        Returns:
            Dictionary containing different report sections
        """
        reports = {}
        
        # 1. Dataset Overview Report
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("SMS DATASET PROCESSING REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"\n📊 DATASET OVERVIEW")
        report_lines.append(f"   Total Messages: {len(self.processed_data)}")
        report_lines.append(f"   Legitimate Messages: {len(self.legitimate_data)}")
        report_lines.append(f"   Spam Messages: {len(self.spam_data)}")
        report_lines.append(f"   Categories: {len(self.processed_data['category'].unique())}")
        
        # 2. Category Distribution Report
        report_lines.append(f"\n📈 CATEGORY DISTRIBUTION")
        category_dist = self.processed_data['category'].value_counts()
        for category, count in category_dist.items():
            percentage = (count / len(self.processed_data)) * 100
            report_lines.append(f"   {category}: {count} ({percentage:.1f}%)")
        
        # 3. Sample Messages per Category
        report_lines.append(f"\n📝 SAMPLE MESSAGES PER CATEGORY")
        for category in sorted(self.processed_data['category'].unique()):
            report_lines.append(f"\n   [{category.upper()}]")
            samples = self.processed_data[self.processed_data['category'] == category].head(2)
            for idx, row in samples.iterrows():
                truncated_text = row['message_text'][:80] + "..." if len(row['message_text']) > 80 else row['message_text']
                report_lines.append(f"   • {truncated_text}")
        
        reports['dataset_overview'] = "\n".join(report_lines)
        
        # 4. Clustering Report
        if self.clusterer:
            cluster_report = self.clusterer.get_clustering_report()
            reports['clustering_analysis'] = cluster_report
        
        # 5. Balancing Report (if balanced)
        if self.balanced_data is not None:
            balance_lines = []
            balance_lines.append("=" * 60)
            balance_lines.append("DATASET BALANCING REPORT")
            balance_lines.append("=" * 60)
            
            balance_lines.append("\n📊 BEFORE BALANCING:")
            original_counts = self.processed_data['category'].value_counts()
            for cat, count in original_counts.items():
                balance_lines.append(f"   {cat}: {count}")
            
            balance_lines.append("\n📊 AFTER BALANCING:")
            balanced_counts = self.balanced_data['category'].value_counts()
            for cat, count in balanced_counts.items():
                balance_lines.append(f"   {cat}: {count}")
            
            balance_lines.append(f"\n📈 BALANCING STATISTICS:")
            balance_lines.append(f"   Total samples added: {len(self.balanced_data) - len(self.processed_data)}")
            balance_lines.append(f"   Balance improvement: {balanced_counts.std() / balanced_counts.mean():.2%}")
            
            reports['balancing_report'] = "\n".join(balance_lines)
        
        return reports