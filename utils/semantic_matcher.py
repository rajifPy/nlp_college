from utils.model_loader import ModelLoader
import numpy as np

class SemanticMatcher:
    """Enhanced semantic matcher menggunakan model Anda"""
    
    def __init__(self, model_loader: ModelLoader = None):
        self.model_loader = model_loader or ModelLoader()
    
    def compute_similarities(self, document_text: str) -> list:
        """
        Enhanced semantic matching menggunakan model terlatih Anda
        """
        if not self.model_loader.sdg_model:
            print("❌ SDG model not available, using fallback")
            return []
        
        try:
            # Gunakan model Anda untuk prediksi langsung
            results = self.model_loader.predict_sdgs(document_text, top_k=5)
            
            # Format hasil untuk compatibility dengan kode existing
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'sdg_number': result['sdg_number'],
                    'sdg_name': result['sdg_name'],
                    'avg_similarity': result['confidence'],
                    'max_similarity': result['confidence'],
                    'top_indicators': [{
                        'indicator': 'Model-based prediction',
                        'similarity': result['confidence']
                    }],
                    'explanation': result['explanation'],
                    'matched_keywords': result['matched_keywords']
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error in semantic matching: {str(e)}")
            return []
    
    def get_detailed_analysis(self, document_text: str) -> dict:
        """
        Analisis detail menggunakan model Anda
        """
        results = self.model_loader.predict_sdgs(document_text, top_k=3)
        
        return {
            'top_3_sdgs': results,
            'confidence_levels': self._get_confidence_levels(results),
            'matched_keywords': self._get_all_keywords(results),
            'explanation': self._generate_overall_explanation(results)
        }
    
    def _get_confidence_levels(self, results):
        levels = {'high': 0, 'medium': 0, 'low': 0}
        for result in results:
            if result['confidence'] >= 0.7:
                levels['high'] += 1
            elif result['confidence'] >= 0.4:
                levels['medium'] += 1
            else:
                levels['low'] += 1
        return levels
    
    def _get_all_keywords(self, results):
        keywords = []
        for result in results:
            keywords.extend(result.get('matched_keywords', []))
        return list(set(keywords))[:20]
    
    def _generate_overall_explanation(self, results):
        if not results:
            return "No SDGs detected with sufficient confidence."
        
        sdg_names = [f"SDG {r['sdg_number']}: {r['sdg_name']}" for r in results]
        confidences = [f"{r['confidence']:.2f}" for r in results]
        
        return (f"Top SDGs detected: {', '.join(sdg_names)}. "
                f"Confidence scores: {', '.join(confidences)}. "
                f"Analysis powered by trained machine learning model.")