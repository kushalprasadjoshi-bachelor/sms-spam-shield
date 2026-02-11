"""
Text preprocessing utilities for SMS messages.
Preserves original messages while creating cleaned versions for clustering.
"""
import re
import nltk
from typing import Dict, List, Optional
import pandas as pd

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

class TextPreprocessor:
    """Preprocess SMS text for clustering while preserving original messages."""
    
    def __init__(self):
        """Initialize the text preprocessor."""
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        
        # Common SMS abbreviations mapping
        self.sms_abbreviations = {
            'u': 'you', 'r': 'are', 'btw': 'by the way',
            'lol': 'laugh out loud', 'brb': 'be right back',
            'ttyl': 'talk to you later', 'idk': "i don't know",
            'omg': 'oh my god', 'wtf': 'what the fuck',
            'imo': 'in my opinion', 'smh': 'shake my head',
            'afaik': 'as far as i know', 'tbh': 'to be honest',
            'fyi': 'for your information', 'np': 'no problem',
            'thx': 'thanks', 'pls': 'please', 'msg': 'message',
            'txt': 'text', '&': 'and', '+': 'and',
            'w/': 'with', 'w/o': 'without'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for clustering.
        Preserves the original message separately.
        
        Args:
            text: Original SMS text
            
        Returns:
            Cleaned text for clustering
        """
        if not isinstance(text, str):
            return ""
        
        # Store original
        original_text = text
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace SMS abbreviations
        for abbr, full in self.sms_abbreviations.items():
            text = re.sub(rf'\b{abbr}\b', full, text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Remove phone numbers (preserve context but remove actual numbers)
        text = re.sub(r'\b\d{10,}\b', 'phonenumber', text)
        
        # Remove currency symbols and amounts
        text = re.sub(r'[$€£¥]\s*\d+', 'moneyamount', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_for_clustering(self, text: str) -> str:
        """
        Complete preprocessing pipeline for clustering.
        
        Args:
            text: Original SMS text
            
        Returns:
            Fully preprocessed text ready for vectorization
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = nltk.word_tokenize(cleaned)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts for clustering.
        
        Args:
            texts: List of original SMS texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess_for_clustering(text) for text in texts]
    
    def preserve_original_with_clean(self, df: pd.DataFrame, 
                                    text_column: str = 'sms_text') -> pd.DataFrame:
        """
        Create a dataframe with both original and cleaned text.
        
        Args:
            df: DataFrame containing SMS messages
            text_column: Name of the column containing text
            
        Returns:
            DataFrame with original_text and cleaned_text columns
        """
        result_df = df.copy()
        
        # Preserve original text
        result_df['original_text'] = result_df[text_column]
        
        # Create cleaned version for clustering
        result_df['cleaned_text'] = result_df[text_column].apply(
            self.preprocess_for_clustering
        )
        
        return result_df
    
    def analyze_text_features(self, texts: List[str]) -> Dict:
        """
        Analyze text features for reporting.
        
        Args:
            texts: List of SMS texts
            
        Returns:
            Dictionary with text analysis metrics
        """
        analysis = {
            'total_messages': len(texts),
            'total_characters': sum(len(text) for text in texts),
            'total_words': sum(len(text.split()) for text in texts),
            'avg_message_length': sum(len(text) for text in texts) / len(texts) if texts else 0,
            'avg_words_per_message': sum(len(text.split()) for text in texts) / len(texts) if texts else 0,
            'unique_words': len(set(' '.join(texts).split())),
        }
        
        # Find most common words
        all_words = ' '.join(texts).lower().split()
        word_freq = nltk.FreqDist(all_words)
        analysis['top_10_words'] = word_freq.most_common(10)
        
        return analysis