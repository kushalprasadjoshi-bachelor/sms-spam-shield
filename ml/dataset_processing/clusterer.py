"""
Spam subtype clustering module using unsupervised learning.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class SpamSubtypeClusterer:
    """Clusters spam messages into subtypes using unsupervised learning."""
    
    def __init__(self, config: Dict):
        """
        Initialize the spam subtype clusterer.
        
        Args:
            config: Clustering configuration dictionary
        """
        self.config = config
        self.vectorizer = None
        self.cluster_model = None
        self.n_clusters = None
        self.cluster_labels = None
        
    def fit_predict(self, texts: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit clustering model and predict clusters for texts.
        
        Args:
            texts: List of spam message texts
            
        Returns:
            Tuple of (cluster_labels, cluster_centers)
        """
        # Step 1: Vectorize text using TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.get('max_features', 5000),
            max_df=self.config.get('max_df', 0.95),
            min_df=self.config.get('min_df', 2),
            stop_words='english'
        )
        
        X = self.vectorizer.fit_transform(texts)
        print(f"  ✓ Vectorized {len(texts)} messages with {X.shape[1]} features")
        
        # Step 2: Determine optimal number of clusters if not specified
        if self.config['method'] == 'kmeans':
            if 'n_clusters' in self.config['kmeans']:
                self.n_clusters = self.config['kmeans']['n_clusters']
            else:
                self.n_clusters = self._find_optimal_clusters(X)
            
            print(f"  ✓ Using {self.n_clusters} clusters for KMeans")
            
            # Train KMeans
            self.cluster_model = KMeans(
                n_clusters=self.n_clusters,
                n_init=self.config['kmeans'].get('n_init', 10),
                max_iter=self.config['kmeans'].get('max_iter', 300),
                random_state=42
            )
            
        elif self.config['method'] == 'dbscan':
            self.cluster_model = DBSCAN(
                eps=self.config['dbscan'].get('eps', 0.5),
                min_samples=self.config['dbscan'].get('min_samples', 5)
            )
            
        elif self.config['method'] == 'hierarchical':
            self.n_clusters = self.config['hierarchical'].get('n_clusters', 4)
            self.cluster_model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage=self.config['hierarchical'].get('linkage', 'ward')
            )
        
        # Step 3: Fit and predict
        self.cluster_labels = self.cluster_model.fit_predict(X)
        
        # For DBSCAN, noise points are labeled as -1
        if self.config['method'] == 'dbscan':
            unique_clusters = set(self.cluster_labels)
            if -1 in unique_clusters:
                print(f"  ⚠️  DBSCAN identified {sum(self.cluster_labels == -1)} noise points")
        
        # Step 4: Get cluster centers (if available)
        cluster_centers = None
        if hasattr(self.cluster_model, 'cluster_centers_'):
            cluster_centers = self.cluster_model.cluster_centers_
        
        # Step 5: Evaluate clustering quality
        if len(set(self.cluster_labels)) > 1:
            self._evaluate_clustering(X)
        
        return self.cluster_labels, cluster_centers
    
    def _find_optimal_clusters(self, X, max_clusters: int = 10) -> int:
        """
        Find optimal number of clusters using silhouette score.
        
        Args:
            X: Feature matrix
            max_clusters: Maximum number of clusters to try
            
        Returns:
            Optimal number of clusters
        """
        silhouette_scores = []
        
        for n in range(2, min(max_clusters + 1, X.shape[0])):
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            if len(set(cluster_labels)) > 1:
                score = silhouette_score(X, cluster_labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)
        
        if silhouette_scores:
            optimal_n = np.argmax(silhouette_scores) + 2
            print(f"  ✓ Optimal clusters found: {optimal_n} (silhouette score: {max(silhouette_scores):.3f})")
            return optimal_n
        else:
            print("  ⚠️  Could not determine optimal clusters, using default 4")
            return 4
    
    def _evaluate_clustering(self, X) -> None:
        """
        Evaluate clustering quality using multiple metrics.
        
        Args:
            X: Feature matrix
        """
        if len(set(self.cluster_labels)) <= 1:
            print("  ⚠️  Only one cluster found, skipping evaluation")
            return
        
        try:
            # Silhouette Score (-1 to 1, higher is better)
            silhouette = silhouette_score(X, self.cluster_labels)
            
            # Calinski-Harabasz Score (higher is better)
            calinski = calinski_harabasz_score(X.toarray(), self.cluster_labels)
            
            print(f"  📊 Clustering Evaluation:")
            print(f"    • Silhouette Score: {silhouette:.3f}")
            print(f"    • Calinski-Harabasz Score: {calinski:.2f}")
            
            # Interpretation
            if silhouette > 0.7:
                print(f"    • Interpretation: Strong structure found")
            elif silhouette > 0.5:
                print(f"    • Interpretation: Reasonable structure")
            elif silhouette > 0.25:
                print(f"    • Interpretation: Weak structure")
            else:
                print(f"    • Interpretation: No substantial structure")
                
        except Exception as e:
            print(f"  ⚠️  Could not compute clustering metrics: {e}")
    
    def get_cluster_keywords(self, top_n: int = 10) -> Dict[int, List[str]]:
        """
        Get top keywords for each cluster.
        
        Args:
            top_n: Number of top keywords to retrieve
            
        Returns:
            Dictionary mapping cluster_id to list of keywords
        """
        if self.cluster_model is None or self.vectorizer is None:
            raise ValueError("Clusterer has not been fitted yet")
        
        if not hasattr(self.cluster_model, 'cluster_centers_'):
            raise ValueError("Cluster model does not have cluster centers")
        
        feature_names = self.vectorizer.get_feature_names_out()
        cluster_keywords = {}
        
        for cluster_id in range(self.n_clusters):
            # Get the cluster center
            center = self.cluster_model.cluster_centers_[cluster_id]
            
            # Get indices of top features
            top_indices = center.argsort()[-top_n:][::-1]
            
            # Get corresponding feature names
            keywords = [feature_names[i] for i in top_indices]
            cluster_keywords[cluster_id] = keywords
        
        return cluster_keywords
    
    def get_clustering_report(self) -> str:
        """Generate a comprehensive clustering report."""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("CLUSTERING ANALYSIS REPORT")
        report_lines.append("=" * 60)
        
        report_lines.append(f"\n🔧 CLUSTERING CONFIGURATION:")
        report_lines.append(f"   Method: {self.config['method']}")
        
        if self.config['method'] == 'kmeans':
            report_lines.append(f"   Number of clusters: {self.n_clusters}")
            report_lines.append(f"   n_init: {self.config['kmeans'].get('n_init', 10)}")
            report_lines.append(f"   max_iter: {self.config['kmeans'].get('max_iter', 300)}")
        
        report_lines.append(f"\n📊 CLUSTER DISTRIBUTION:")
        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        
        for label, count in zip(unique_labels, counts):
            if label == -1:
                report_lines.append(f"   Noise points: {count}")
            else:
                percentage = (count / len(self.cluster_labels)) * 100
                report_lines.append(f"   Cluster {label}: {count} messages ({percentage:.1f}%)")
        
        # Get cluster keywords if available
        try:
            cluster_keywords = self.get_cluster_keywords()
            report_lines.append(f"\n🔑 TOP KEYWORDS PER CLUSTER:")
            
            for cluster_id, keywords in cluster_keywords.items():
                report_lines.append(f"\n   Cluster {cluster_id}:")
                report_lines.append(f"     {', '.join(keywords)}")
                
        except Exception as e:
            report_lines.append(f"\n⚠️  Could not extract cluster keywords: {e}")
        
        return "\n".join(report_lines)