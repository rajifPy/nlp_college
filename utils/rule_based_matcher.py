import pandas as pd
import re
from typing import Dict, List, Tuple

class RuleBasedMatcher:
    """Detects SDG indicators using curated inclusion/exclusion patterns."""
    
    def __init__(self, patterns_path: str = 'data/sdg_patterns.csv'):
        self.patterns_df = self._load_patterns(patterns_path)
    
    def _load_patterns(self, patterns_path: str) -> pd.DataFrame:
        """Load SDG patterns from CSV file."""
        try:
            df = pd.read_csv(patterns_path)
            return df
        except FileNotFoundError:
            print(f"Warning: Pattern file {patterns_path} not found. Using default patterns.")
            return self._create_default_patterns()
    
    def _create_default_patterns(self) -> pd.DataFrame:
        """Create default SDG patterns."""
        patterns = []
        
        # SDG 1: No Poverty
        patterns.append({
            'sdg_number': 1,
            'sdg_name': 'No Poverty',
            'inclusion_patterns': ['poverty', 'poor', 'low income', 'financial hardship', 'economic disadvantage', 'income inequality'],
            'exclusion_patterns': ['rich', 'wealthy', 'high income', 'luxury']
        })
        
        # SDG 2: Zero Hunger
        patterns.append({
            'sdg_number': 2,
            'sdg_name': 'Zero Hunger',
            'inclusion_patterns': ['hunger', 'food security', 'malnutrition', 'famine', 'food shortage', 'agriculture', 'crop production'],
            'exclusion_patterns': ['food waste', 'overconsumption', 'obesity']
        })
        
        # SDG 3: Good Health and Well-being
        patterns.append({
            'sdg_number': 3,
            'sdg_name': 'Good Health and Well-being',
            'inclusion_patterns': ['health', 'well-being', 'disease prevention', 'healthcare', 'medical', 'vaccination', 'mental health', 'hygiene'],
            'exclusion_patterns': ['disease outbreak', 'pandemic', 'health crisis']
        })
        
        # Add more SDGs as needed...
        
        return pd.DataFrame(patterns)
    
    def match_sdgs(self, text: str) -> List[Dict]:
        """Match SDGs using rule-based patterns."""
        results = []
        text_lower = text.lower()
        
        for _, row in self.patterns_df.iterrows():
            inclusion_patterns = row['inclusion_patterns'].split(',') if isinstance(row['inclusion_patterns'], str) else []
            exclusion_patterns = row['exclusion_patterns'].split(',') if isinstance(row['exclusion_patterns'], str) else []
            
            # Calculate matches
            inclusion_matches = [pattern for pattern in inclusion_patterns 
                               if re.search(r'\b' + re.escape(pattern.strip()) + r'\b', text_lower)]
            
            exclusion_matches = [pattern for pattern in exclusion_patterns 
                               if re.search(r'\b' + re.escape(pattern.strip()) + r'\b', text_lower)]
            
            # Calculate score based on matches
            inclusion_score = len(inclusion_matches) * 0.3
            exclusion_score = len(exclusion_matches) * 0.5
            
            confidence = max(0.0, inclusion_score - exclusion_score)
            
            if confidence > 0.1:  # Threshold for inclusion
                results.append({
                    'sdg_number': row['sdg_number'],
                    'sdg_name': row['sdg_name'],
                    'confidence': confidence,
                    'matched_keywords': inclusion_matches,
                    'excluded_keywords': exclusion_matches,
                    'inclusion_scope': 'broad' if len(inclusion_matches) > 2 else 'narrow'
                })
        
        # Sort by confidence and return top results
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:10]  # Return top 10 matches