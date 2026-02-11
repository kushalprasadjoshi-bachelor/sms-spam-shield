# ml/data_processing/validator.py
"""
Dataset validation and quality checking utilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re

class DatasetValidator:
    """Validates dataset quality and consistency."""
    
    @staticmethod
    def validate_dataset(df: pd.DataFrame, 
                        text_column: str = 'sms_text',
                        label_column: str = 'category') -> Dict:
        """
        Validate dataset quality.
        
        Args:
            df: DataFrame to validate
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check 1: Required columns exist
        required_columns = [text_column, label_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(
                f"Missing required columns: {missing_columns}"
            )
        
        # Check 2: No empty texts
        empty_texts = df[text_column].isna() | (df[text_column].str.strip() == '')
        if empty_texts.any():
            validation_results['warnings'].append(
                f"Found {empty_texts.sum()} empty text messages"
            )
        
        # Check 3: No NaN labels
        nan_labels = df[label_column].isna()
        if nan_labels.any():
            validation_results['is_valid'] = False
            validation_results['issues'].append(
                f"Found {nan_labels.sum()} NaN labels"
            )
        
        # Check 4: Text length distribution
        text_lengths = df[text_column].str.len()
        validation_results['statistics']['text_length'] = {
            'min': int(text_lengths.min()),
            'max': int(text_lengths.max()),
            'mean': float(text_lengths.mean()),
            'std': float(text_lengths.std())
        }
        
        # Check 5: Label distribution
        label_distribution = df[label_column].value_counts().to_dict()
        validation_results['statistics']['label_distribution'] = label_distribution
        
        # Check 6: Duplicate messages
        duplicate_texts = df[text_column].duplicated().sum()
        if duplicate_texts > 0:
            validation_results['warnings'].append(
                f"Found {duplicate_texts} duplicate messages"
            )
        
        # Check 7: Label consistency
        unique_labels = df[label_column].nunique()
        validation_results['statistics']['unique_labels'] = unique_labels
        
        # Check 8: Character encoding issues
        encoding_issues = 0
        for text in df[text_column]:
            try:
                text.encode('utf-8')
            except UnicodeEncodeError:
                encoding_issues += 1
        
        if encoding_issues > 0:
            validation_results['warnings'].append(
                f"Found {encoding_issues} potential encoding issues"
            )
        
        return validation_results
    
    @staticmethod
    def validate_category_labels(categories: List[str], 
                               expected_categories: List[str] = None) -> Dict:
        """
        Validate category labels.
        
        Args:
            categories: List of category labels
            expected_categories: Expected categories (optional)
            
        Returns:
            Validation results
        """
        results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        unique_categories = set(categories)
        results['statistics']['unique_categories'] = len(unique_categories)
        results['statistics']['category_list'] = sorted(list(unique_categories))
        
        # Check for unexpected categories
        if expected_categories:
            unexpected = unique_categories - set(expected_categories)
            if unexpected:
                results['warnings'] = [f"Unexpected categories: {unexpected}"]
        
        # Check category naming consistency
        invalid_categories = []
        for category in unique_categories:
            if not isinstance(category, str):
                invalid_categories.append(category)
            elif re.search(r'[^\w\s-]', category):
                results['warnings'].append(
                    f"Category '{category}' contains special characters"
                )
        
        if invalid_categories:
            results['issues'].append(
                f"Invalid category types: {invalid_categories}"
            )
            results['is_valid'] = False
        
        return results
    
    @staticmethod
    def generate_validation_report(df: pd.DataFrame,
                                 text_column: str = 'sms_text',
                                 label_column: str = 'category') -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            df: DataFrame to validate
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Validation report as string
        """
        validation = DatasetValidator.validate_dataset(df, text_column, label_column)
        
        report = []
        report.append("=" * 60)
        report.append("DATASET VALIDATION REPORT")
        report.append("=" * 60)
        
        report.append(f"\n📊 OVERVIEW:")
        report.append(f"  Total samples: {len(df)}")
        report.append(f"  Columns: {list(df.columns)}")
        
        report.append(f"\n✅ VALIDATION STATUS:")
        if validation['is_valid']:
            report.append("  ✓ Dataset is valid")
        else:
            report.append("  ✗ Dataset has issues")
        
        if validation['issues']:
            report.append(f"\n❌ ISSUES FOUND:")
            for issue in validation['issues']:
                report.append(f"  • {issue}")
        
        if validation['warnings']:
            report.append(f"\n⚠️  WARNINGS:")
            for warning in validation['warnings']:
                report.append(f"  • {warning}")
        
        report.append(f"\n📈 STATISTICS:")
        
        # Text statistics
        text_stats = validation['statistics'].get('text_length', {})
        report.append(f"\n  TEXT LENGTH:")
        report.append(f"    Min: {text_stats.get('min', 'N/A')} characters")
        report.append(f"    Max: {text_stats.get('max', 'N/A')} characters")
        report.append(f"    Mean: {text_stats.get('mean', 'N/A'):.1f} characters")
        report.append(f"    Std: {text_stats.get('std', 'N/A'):.1f} characters")
        
        # Label statistics
        label_stats = validation['statistics'].get('label_distribution', {})
        report.append(f"\n  LABEL DISTRIBUTION:")
        for label, count in label_stats.items():
            percentage = (count / len(df)) * 100
            report.append(f"    {label}: {count} ({percentage:.1f}%)")
        
        report.append(f"\n  UNIQUE LABELS: {validation['statistics'].get('unique_labels', 'N/A')}")
        
        return "\n".join(report)