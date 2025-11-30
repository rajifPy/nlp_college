import joblib
import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import re

class ModelLoader:
    """Loader untuk model SDG classifier (.joblib format)"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or os.getenv('MODEL_PATH', './models')
        self.sdg_model = None
        self.sbert_model = None
        self.sdg_labels = [
            'SDG 1: No Poverty', 'SDG 2: Zero Hunger', 'SDG 3: Good Health and Well-being',
            'SDG 4: Quality Education', 'SDG 5: Gender Equality', 'SDG 6: Clean Water and Sanitation',
            'SDG 7: Affordable and Clean Energy', 'SDG 8: Decent Work and Economic Growth',
            'SDG 9: Industry, Innovation and Infrastructure', 'SDG 10: Reduced Inequality',
            'SDG 11: Sustainable Cities and Communities', 'SDG 12: Responsible Consumption and Production',
            'SDG 13: Climate Action', 'SDG 14: Life Below Water', 'SDG 15: Life on Land',
            'SDG 16: Peace, Justice and Strong Institutions', 'SDG 17: Partnerships for the Goals'
        ]
        
        self.load_models()
    
    def load_models(self):
        """Load model .joblib dari path yang ditentukan"""
        try:
            print(f"🔍 Looking for model in: {self.model_path}")
            
            # Load SDG classifier model (.joblib)
            sdg_model_path = Path(self.model_path) / 'sdg_classifier.joblib'
            if sdg_model_path.exists():
                print(f"📥 Loading model from: {sdg_model_path}")
                self.sdg_model = joblib.load(sdg_model_path)
                print(f"✅ SDG classifier model loaded successfully! Model type: {type(self.sdg_model)}")
            else:
                print(f"❌ Model file not found at: {sdg_model_path}")
                print("💡 Please place your trained .joblib model at 'models/sdg_classifier.joblib'")
                self.sdg_model = None
            
            # Load Sentence-BERT model untuk semantic matching
            print("📥 Loading Sentence-BERT model...")
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence-BERT model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict_sdgs(self, text: str, top_k: int = 3):
        """
        Prediksi SDGs menggunakan model .joblib Anda
        
        Args:
            text: Input text dari dokumen
            top_k: Jumlah top SDGs yang ingin ditampilkan
            
        Returns:
            List of dictionaries dengan format:
            [
                {
                    'sdg_number': 1,
                    'sdg_name': 'No Poverty',
                    'confidence': 0.95,
                    'matched_keywords': ['poverty', 'low income'],
                    'explanation': 'High confidence match based on economic terms'
                }
            ]
        """
        if not self.sdg_model:
            print("❌ SDG model not loaded, using fallback rule-based matching")
            return self._fallback_prediction(text, top_k)
        
        try:
            print(f"🔍 Analyzing text: {text[:100]}...")
            
            # Preprocessing text
            processed_text = self._preprocess_text(text)
            print(f"🧹 Processed text: {processed_text[:100]}...")
            
            # Get model predictions
            print("🧠 Getting model predictions...")
            probabilities = self.sdg_model.predict_proba([processed_text])[0]
            print(f"📈 Raw probabilities: {probabilities}")
            
            # Get top-k indices
            top_indices = np.argsort(probabilities)[::-1][:top_k]
            print(f"⭐ Top indices: {top_indices}, Top probabilities: {[probabilities[i] for i in top_indices]}")
            
            results = []
            for idx in top_indices:
                prob = probabilities[idx]
                if prob < 0.1:  # Threshold confidence
                    print(f"⏭️ Skipping SDG {idx+1} with low confidence: {prob:.3f}")
                    continue
                
                sdg_name = self.sdg_labels[idx]
                sdg_number = idx + 1
                
                print(f"🎯 SDG {sdg_number}: {sdg_name} - Confidence: {prob:.3f}")
                
                # Generate explanation
                explanation = self._generate_explanation(sdg_number, prob, text)
                
                results.append({
                    'sdg_number': sdg_number,
                    'sdg_name': sdg_name.split(': ', 1)[1],
                    'confidence': float(prob),
                    'matched_keywords': self._extract_keywords(text, sdg_number),
                    'explanation': explanation
                })
            
            print(f"✅ Prediction complete. Found {len(results)} relevant SDGs")
            return results
            
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._fallback_prediction(text, top_k)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocessing text sesuai dengan training"""
        try:
            # Lowercase
            text = text.lower()
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            print(f"⚠️ Error in preprocessing: {str(e)}")
            return text
    
    def _generate_explanation(self, sdg_number: int, confidence: float, text: str) -> str:
        """Generate penjelasan berdasarkan confidence dan keywords"""
        explanation_templates = {
            'high': "Strong match with relevant terminology and context",
            'medium': "Moderate match with some relevant indicators present",
            'low': "Weak match, may require manual verification"
        }
        
        if confidence >= 0.7:
            level = 'high'
        elif confidence >= 0.4:
            level = 'medium'
        else:
            level = 'low'
        
        keywords = self._extract_keywords(text, sdg_number)[:3]
        keyword_str = ', '.join(keywords) if keywords else "general terms"
        
        return f"{explanation_templates[level]}. Key terms: {keyword_str}."
    
    def _extract_keywords(self, text: str, sdg_number: int) -> list:
        """Extract relevant keywords untuk SDG tertentu"""
        sdg_keywords = {
            1: ['poverty', 'poor', 'low income', 'financial hardship', 'economic disadvantage'],
            2: ['hunger', 'food security', 'malnutrition', 'famine', 'agriculture', 'crop'],
            3: ['health', 'well-being', 'disease', 'medical', 'healthcare', 'hygiene', 'mental health'],
            4: ['education', 'school', 'learning', 'literacy', 'educational', 'student', 'teacher'],
            5: ['gender', 'equality', 'women', 'female', 'discrimination', 'empowerment'],
            6: ['water', 'sanitation', 'clean water', 'hygiene', 'wastewater', 'water quality'],
            7: ['energy', 'renewable', 'solar', 'wind', 'electricity', 'affordable energy'],
            13: ['climate', 'carbon', 'emission', 'global warming', 'sustainability', 'environmental'],
            15: ['forest', 'biodiversity', 'ecosystem', 'wildlife', 'conservation', 'land'],
        }
        
        text_lower = text.lower()
        keywords = sdg_keywords.get(sdg_number, [])
        matched = [kw for kw in keywords if kw in text_lower]
        
        return matched[:5]  # Return max 5 keywords
    
    def _fallback_prediction(self, text: str, top_k: int = 3):
        """Fallback jika model tidak tersedia"""
        print("🔄 Using fallback rule-based prediction")
        
        # Simple rule-based fallback
        fallback_results = []
        text_lower = text.lower()
        
        sdg_matches = {
            1: len([w for w in ['poverty', 'poor', 'low income'] if w in text_lower]),
            2: len([w for w in ['hunger', 'food security', 'malnutrition'] if w in text_lower]),
            3: len([w for w in ['health', 'medical', 'disease', 'well-being'] if w in text_lower]),
            13: len([w for w in ['climate', 'carbon', 'emission', 'global warming'] if w in text_lower]),
            15: len([w for w in ['forest', 'biodiversity', 'ecosystem', 'wildlife'] if w in text_lower]),
        }
        
        sorted_sdgs = sorted(sdg_matches.items(), key=lambda x: x[1], reverse=True)
        
        for sdg_num, count in sorted_sdgs[:top_k]:
            if count > 0:
                fallback_results.append({
                    'sdg_number': sdg_num,
                    'sdg_name': self.sdg_labels[sdg_num-1].split(': ', 1)[1],
                    'confidence': min(0.9, count * 0.3),
                    'matched_keywords': [w for w in ['poverty', 'food security', 'health', 'climate', 'forest'] if w in text_lower],
                    'explanation': 'Fallback rule-based prediction - please train and place your model in models/sdg_classifier.joblib'
                })
        
        if not fallback_results:
            fallback_results = [{
                'sdg_number': 0,
                'sdg_name': 'No SDGs Detected',
                'confidence': 0.0,
                'matched_keywords': [],
                'explanation': 'No relevant SDGs detected. Please ensure your trained model is placed in the models folder.'
            }]
        
        return fallback_results