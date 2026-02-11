"""
Dataset balancing utilities using SMOTE and other techniques.
"""
import numpy as np
from typing import Dict, Tuple, List
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from collections import Counter

class DatasetBalancer:
    """Balances dataset using various techniques."""
    
    def __init__(self, config: Dict):
        """
        Initialize dataset balancer.
        
        Args:
            config: Balancing configuration dictionary
        """
        self.config = config
        self.balancer = None
        
    def balance(self, X: List[str], y: List[int]) -> Tuple[List[str], List[int]]:
        """
        Balance the dataset.
        
        Args:
            X: List of text samples
            y: List of labels
            
        Returns:
            Tuple of balanced (X, y)
        """
        print(f"  ⚖️  Balancing dataset...")
        print(f"    Original class distribution: {dict(Counter(y))}")
        
        # Convert text to numerical features for SMOTE
        # Since SMOTE requires numerical features, we need a simple representation
        # We'll use a simple character-based frequency representation
        
        # Create a simple numerical representation
        X_numerical = self._text_to_numerical(X)
        
        # Apply balancing
        method = self.config['method']
        
        if method == 'smote':
            self.balancer = SMOTE(
                sampling_strategy=self.config['smote'].get('sampling_strategy', 'auto'),
                k_neighbors=self.config['smote'].get('k_neighbors', 5),
                random_state=self.config['smote'].get('random_state', 42)
            )
            
        elif method == 'random_oversample':
            self.balancer = RandomOverSampler(
                sampling_strategy=self.config['random_oversample'].get('sampling_strategy', 'auto'),
                random_state=self.config['random_oversample'].get('random_state', 42)
            )
            
        elif method == 'adasyn':
            self.balancer = ADASYN(
                sampling_strategy=self.config.get('adasyn', {}).get('sampling_strategy', 'auto'),
                random_state=self.config.get('adasyn', {}).get('random_state', 42)
            )
        
        # Resample the data
        X_resampled, y_resampled = self.balancer.fit_resample(X_numerical, y)
        
        # Map numerical features back to text
        # For SMOTE-generated samples, we find the nearest original text
        X_text_resampled = self._numerical_to_text(X_resampled, X, X_numerical, y_resampled)
        
        print(f"    New class distribution: {dict(Counter(y_resampled))}")
        
        return X_text_resampled, y_resampled
    
    def _text_to_numerical(self, texts: List[str]) -> np.ndarray:
        """
        Convert text to numerical features using simple character frequencies.
        
        Args:
            texts: List of text samples
            
        Returns:
            Numerical feature matrix
        """
        # Use a simple bag-of-characters approach
        # This is a simple representation for balancing purposes
        all_chars = set()
        for text in texts:
            all_chars.update(text.lower())
        
        # Create character mapping
        char_list = sorted(list(all_chars))
        char_to_idx = {char: i for i, char in enumerate(char_list)}
        
        # Create feature matrix
        features = np.zeros((len(texts), len(char_list)))
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            for char in text_lower:
                if char in char_to_idx:
                    features[i, char_to_idx[char]] += 1
        
        return features
    
    def _numerical_to_text(self, X_resampled: np.ndarray, 
                          original_texts: List[str],
                          original_features: np.ndarray,
                          y_resampled: List[int]) -> List[str]:
        """
        Map resampled numerical features back to text.
        
        Args:
            X_resampled: Resampled numerical features
            original_texts: Original text samples
            original_features: Original numerical features
            y_resampled: Resampled labels
            
        Returns:
            List of text samples (original + generated)
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Find nearest neighbors for generated samples
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(original_features)
        
        result_texts = []
        
        for i, sample in enumerate(X_resampled):
            if i < len(original_texts):
                # This is an original sample
                result_texts.append(original_texts[i])
            else:
                # This is a generated sample - find nearest original
                distances, indices = nn.kneighbors([sample])
                nearest_idx = indices[0][0]
                result_texts.append(original_texts[nearest_idx])
        
        return result_texts
    
    def get_balancing_report(self, original_y: List[int], 
                           resampled_y: List[int]) -> str:
        """
        Generate balancing report.
        
        Args:
            original_y: Original labels
            resampled_y: Resampled labels
            
        Returns:
            Balancing report as string
        """
        original_counts = Counter(original_y)
        resampled_counts = Counter(resampled_y)
        
        report = []
        report.append("=" * 60)
        report.append("DATASET BALANCING REPORT")
        report.append("=" * 60)
        
        report.append(f"\n📊 ORIGINAL DISTRIBUTION:")
        for label, count in original_counts.items():
            report.append(f"  Class {label}: {count} samples")
        
        report.append(f"\n📊 RESAMPLED DISTRIBUTION:")
        for label, count in resampled_counts.items():
            report.append(f"  Class {label}: {count} samples")
        
        report.append(f"\n📈 BALANCING STATISTICS:")
        report.append(f"  Total original samples: {len(original_y)}")
        report.append(f"  Total resampled samples: {len(resampled_y)}")
        report.append(f"  Samples added: {len(resampled_y) - len(original_y)}")
        
        # Calculate balance metrics
        original_std = np.std(list(original_counts.values()))
        resampled_std = np.std(list(resampled_counts.values()))
        
        report.append(f"  Original std deviation: {original_std:.2f}")
        report.append(f"  Resampled std deviation: {resampled_std:.2f}")
        report.append(f"  Balance improvement: {(original_std - resampled_std)/original_std:.1%}")
        
        return "\n".join(report)