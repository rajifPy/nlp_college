import os
import requests
import joblib
from pathlib import Path

def download_models():
    """Download models saat deployment di Vercel"""
    model_dir = Path('/tmp/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_url = os.getenv('MODEL_URL')
    if not model_url:
        print("⚠️ MODEL_URL not set, skipping model download")
        return
    
    try:
        print(f"⏬ Downloading model from {model_url}")
        response = requests.get(model_url, timeout=300)  # 5 minutes timeout
        
        if response.status_code == 200:
            model_path = model_dir / 'sdg_classifier.joblib'
            with open(model_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Model downloaded successfully to {model_path}")
            
            # Test load model
            joblib.load(model_path)
            print("✅ Model loaded successfully")
        else:
            print(f"❌ Failed to download model. Status code: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")

if __name__ == "__main__":
    download_models()