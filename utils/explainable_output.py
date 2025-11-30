from typing import Dict, List, Tuple
import json

class ExplainableOutput:
    """Generates explainable output with matched/missing keywords and confidence levels."""
    
    def __init__(self):
        self.explanation_templates = {
            'high_confidence': "Strong match with multiple relevant indicators and keywords",
            'medium_confidence': "Moderate match with some relevant indicators",
            'low_confidence': "Weak match, may require manual verification"
        }
    
    def generate_explanation(self, rule_results: List[Dict], semantic_results: List[Dict]) -> Dict:
        """Generate comprehensive explanation combining rule-based and semantic results."""
        
        # Combine results from both methods
        combined_results = self._combine_results(rule_results, semantic_results)
        
        # Get top 3 SDGs
        top_3_sdgs = combined_results[:3]
        
        # Generate detailed explanations
        explanations = []
        
        for sdg in top_3_sdgs:
            explanation = self._generate_sdg_explanation(sdg)
            explanations.append(explanation)
        
        # Generate overall summary
        overall_summary = self._generate_overall_summary(top_3_sdgs)
        
        return {
            'top_3_sdgs': top_3_sdgs,
            'explanations': explanations,
            'overall_summary': overall_summary,
            'confidence_levels': self._get_confidence_levels(top_3_sdgs),
            'matched_keywords': self._get_all_matched_keywords(combined_results),
            'missing_keywords': self._get_missing_keywords(top_3_sdgs)
        }
    
    def _combine_results(self, rule_results: List[Dict], semantic_results: List[Dict]) -> List[Dict]:
        """Combine rule-based and semantic matching results."""
        combined = {}
        
        # Add rule-based results
        for result in rule_results:
            sdg_key = f"SDG{result['sdg_number']}"
            if sdg_key not in combined:
                combined[sdg_key] = {
                    'sdg_number': result['sdg_number'],
                    'sdg_name': result['sdg_name'],
                    'rule_confidence': result['confidence'],
                    'semantic_confidence': 0.0,
                    'matched_keywords': result['matched_keywords'],
                    'excluded_keywords': result['excluded_keywords'],
                    'inclusion_scope': result['inclusion_scope']
                }
            else:
                combined[sdg_key]['rule_confidence'] = max(combined[sdg_key]['rule_confidence'], result['confidence'])
                combined[sdg_key]['matched_keywords'].extend(result['matched_keywords'])
        
        # Add semantic results
        for result in semantic_results:
            sdg_key = f"SDG{result['sdg_number']}"
            if sdg_key not in combined:
                combined[sdg_key] = {
                    'sdg_number': result['sdg_number'],
                    'sdg_name': result['sdg_name'],
                    'rule_confidence': 0.0,
                    'semantic_confidence': result['avg_similarity'],
                    'matched_keywords': [],
                    'excluded_keywords': [],
                    'inclusion_scope': 'unknown'
                }
            else:
                combined[sdg_key]['semantic_confidence'] = result['avg_similarity']
        
        # Calculate combined confidence
        for sdg_key, sdg_data in combined.items():
            # Weighted average of rule-based and semantic confidence
            combined_confidence = (sdg_data['rule_confidence'] * 0.4 + 
                                 sdg_data['semantic_confidence'] * 0.6)
            
            sdg_data['combined_confidence'] = combined_confidence
            
            # Remove duplicates from keywords
            sdg_data['matched_keywords'] = list(set(sdg_data['matched_keywords']))
        
        # Convert to list and sort by combined confidence
        result_list = list(combined.values())
        result_list.sort(key=lambda x: x['combined_confidence'], reverse=True)
        
        return result_list
    
    def _generate_sdg_explanation(self, sdg: Dict) -> Dict:
        """Generate explanation for a single SDG."""
        confidence_level = self._determine_confidence_level(sdg['combined_confidence'])
        
        explanation = {
            'sdg_number': sdg['sdg_number'],
            'sdg_name': sdg['sdg_name'],
            'confidence_level': confidence_level,
            'confidence_score': round(sdg['combined_confidence'], 3),
            'matched_keywords': sdg.get('matched_keywords', []),
            'excluded_keywords': sdg.get('excluded_keywords', []),
            'inclusion_scope': sdg.get('inclusion_scope', 'unknown'),
            'detailed_explanation': self._get_detailed_explanation(sdg, confidence_level)
        }
        
        return explanation
    
    def _determine_confidence_level(self, confidence_score: float) -> str:
        """Determine confidence level based on score."""
        if confidence_score >= 0.7:
            return 'high'
        elif confidence_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _get_detailed_explanation(self, sdg: Dict, confidence_level: str) -> str:
        """Generate detailed explanation text."""
        template = self.explanation_templates.get(f"{confidence_level}_confidence", "")
        
        keyword_info = ""
        if sdg.get('matched_keywords'):
            keyword_info = f"Matched keywords: {', '.join(sdg['matched_keywords'][:5])}"
        
        scope_info = f"Inclusion scope: {sdg.get('inclusion_scope', 'unknown').title()}"
        
        return f"{template} {keyword_info} {scope_info}"
    
    def _generate_overall_summary(self, top_3_sdgs: List[Dict]) -> str:
        """Generate overall summary of the analysis."""
        if not top_3_sdgs:
            return "No relevant SDGs detected in the document."
        
        sdg_names = [f"SDG {sdg['sdg_number']}: {sdg['sdg_name']}" for sdg in top_3_sdgs]
        confidence_scores = [f"{sdg['combined_confidence']:.2f}" for sdg in top_3_sdgs]
        
        return (f"Top 3 detected SDGs: {', '.join(sdg_names)}. "
                f"Confidence scores: {', '.join(confidence_scores)}. "
                f"Analysis combines rule-based pattern matching and semantic similarity using Sentence-BERT embeddings.")
    
    def _get_confidence_levels(self, top_3_sdgs: List[Dict]) -> Dict[str, float]:
        """Get confidence levels for visualization."""
        levels = {'high': 0, 'medium': 0, 'low': 0}
        
        for sdg in top_3_sdgs:
            level = self._determine_confidence_level(sdg['combined_confidence'])
            levels[level] += 1
        
        return levels
    
    def _get_all_matched_keywords(self, combined_results: List[Dict]) -> List[str]:
        """Get all matched keywords across all SDGs."""
        all_keywords = []
        for sdg in combined_results:
            all_keywords.extend(sdg.get('matched_keywords', []))
        return list(set(all_keywords))[:20]  # Limit to 20 unique keywords
    
    def _get_missing_keywords(self, top_3_sdgs: List[Dict]) -> List[str]:
        """Get keywords that would strengthen the SDG match."""
        missing_keywords = []
        
        # Common SDG-related terms that might be missing
        important_terms = [
            'sustainable', 'development', 'poverty', 'hunger', 'health', 'education',
            'gender', 'water', 'energy', 'economic', 'industry', 'inequality',
            'cities', 'consumption', 'climate', 'marine', 'land', 'peace', 'partnership'
        ]
        
        return important_terms[:10]  # Placeholder