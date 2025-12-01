from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import sys
import tempfile
from pathlib import Path
from transformers import pipeline
import re

app = Flask(__name__)
CORS(app)

# Initialize zero-shot classification model from HuggingFace
print("🔄 Loading model from HuggingFace...")
try:
    classifier = pipeline("zero-shot-classification", 
                         model="facebook/bart-large-mnli",
                         device=-1)  # CPU mode
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    classifier = None

# SDG Labels
SDG_LABELS = [
    "No Poverty - ending poverty in all forms",
    "Zero Hunger - food security and nutrition",
    "Good Health and Well-being - health and wellness",
    "Quality Education - inclusive education",
    "Gender Equality - women empowerment",
    "Clean Water and Sanitation - water access",
    "Affordable and Clean Energy - renewable energy",
    "Decent Work and Economic Growth - employment",
    "Industry Innovation and Infrastructure - technology",
    "Reduced Inequality - social equality",
    "Sustainable Cities and Communities - urban development",
    "Responsible Consumption and Production - sustainability",
    "Climate Action - climate change mitigation",
    "Life Below Water - ocean conservation",
    "Life on Land - biodiversity protection",
    "Peace Justice and Strong Institutions - governance",
    "Partnerships for the Goals - global cooperation"
]

# Simple PDF text extraction
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF"""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""

def clean_text(text):
    """Clean and preprocess text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def analyze_sdgs(text, top_k=3):
    """Analyze text for SDG alignment using HuggingFace model"""
    if not classifier:
        return [{
            'sdg_number': 1,
            'sdg_name': 'Model Not Available',
            'confidence': 0.0,
            'matched_keywords': [],
            'explanation': 'Model is loading or unavailable'
        }]
    
    try:
        # Limit text length for API
        text = text[:1000] if len(text) > 1000 else text
        
        # Run zero-shot classification
        result = classifier(text, SDG_LABELS, multi_label=True)
        
        # Format results
        results = []
        for i in range(min(top_k, len(result['labels']))):
            label = result['labels'][i]
            score = result['scores'][i]
            
            # Extract SDG number and name
            sdg_parts = label.split(' - ')
            sdg_name = sdg_parts[0]
            sdg_number = i + 1
            
            # Extract keywords (simple approach)
            keywords = extract_keywords(text, label)
            
            results.append({
                'sdg_number': sdg_number,
                'sdg_name': sdg_name,
                'confidence': float(score),
                'matched_keywords': keywords,
                'explanation': f"Analysis based on semantic similarity. Key themes: {', '.join(keywords[:3])}"
            })
        
        return results
    except Exception as e:
        print(f"Error in analysis: {e}")
        return [{
            'sdg_number': 0,
            'sdg_name': 'Analysis Error',
            'confidence': 0.0,
            'matched_keywords': [],
            'explanation': str(e)
        }]

def extract_keywords(text, sdg_label):
    """Extract relevant keywords from text"""
    text_lower = text.lower()
    common_keywords = [
        'poverty', 'hunger', 'health', 'education', 'gender', 'water',
        'energy', 'work', 'innovation', 'inequality', 'cities', 'consumption',
        'climate', 'ocean', 'land', 'peace', 'partnership', 'sustainable',
        'development', 'environment', 'social', 'economic'
    ]
    
    found_keywords = [kw for kw in common_keywords if kw in text_lower]
    return found_keywords[:5]

# HTML Template
HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SDGs Extractor - AI Powered</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { 
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            font-size: 1.1em;
        }
        .badge {
            display: inline-block;
            background: #22c55e;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            margin: 5px auto;
            display: block;
            width: fit-content;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 60px 20px;
            text-align: center;
            background: #f8f9ff;
            transition: all 0.3s;
            cursor: pointer;
            margin: 30px 0;
        }
        .upload-area:hover {
            background: #eef1ff;
            border-color: #764ba2;
        }
        .upload-icon { font-size: 4em; margin-bottom: 20px; }
        input[type="file"] { display: none; }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 25px;
            font-size: 1.1em;
            cursor: pointer;
            margin-top: 20px;
            transition: transform 0.2s;
            display: inline-block;
        }
        .btn:hover { transform: scale(1.05); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        #fileName {
            margin-top: 20px;
            color: #667eea;
            font-weight: bold;
            min-height: 24px;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 30px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .results {
            display: none;
            margin-top: 30px;
        }
        .sdg-card {
            background: #f8f9ff;
            border-left: 5px solid #667eea;
            padding: 20px;
            margin: 15px 0;
            border-radius: 10px;
        }
        .sdg-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .sdg-title {
            font-size: 1.3em;
            color: #333;
            margin: 0;
        }
        .confidence {
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }
        .confidence.high { background: #22c55e; }
        .confidence.medium { background: #f59e0b; }
        .confidence.low { background: #ef4444; }
        .keywords {
            margin-top: 15px;
        }
        .keyword-tag {
            background: #e0e7ff;
            color: #667eea;
            padding: 5px 12px;
            border-radius: 12px;
            font-size: 0.9em;
            margin: 5px 5px 5px 0;
            display: inline-block;
        }
        .explanation {
            color: #666;
            line-height: 1.6;
            margin-top: 10px;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 40px 0;
        }
        .feature {
            text-align: center;
            padding: 20px;
            background: #f8f9ff;
            border-radius: 10px;
        }
        .feature-icon {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #eee;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌱 SDGs Extractor</h1>
        <p class="subtitle">AI-powered Sustainable Development Goals Analysis</p>
        <div class="badge">✨ Powered by HuggingFace Transformers</div>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📄</div>
                <h3>Click to Upload PDF</h3>
                <p>Or drag and drop your file here</p>
                <input type="file" id="fileInput" name="file" accept=".pdf" required>
            </div>
            <div id="fileName"></div>
            <center>
                <button type="submit" class="btn" id="submitBtn">
                    🔍 Analyze Document
                </button>
            </center>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p><strong>Analyzing your document with AI...</strong></p>
            <p style="color: #666; margin-top: 10px;">This may take 30-60 seconds</p>
        </div>
        
        <div class="results" id="results"></div>
        
        <div class="features">
            <div class="feature">
                <div class="feature-icon">🤖</div>
                <h4>AI-Powered</h4>
                <p>Uses HuggingFace BART model</p>
            </div>
            <div class="feature">
                <div class="feature-icon">⚡</div>
                <h4>Fast Analysis</h4>
                <p>Results in under a minute</p>
            </div>
            <div class="feature">
                <div class="feature-icon">🎯</div>
                <h4>Accurate</h4>
                <p>Top 3 relevant SDGs</p>
            </div>
            <div class="feature">
                <div class="feature-icon">🔒</div>
                <h4>Secure</h4>
                <p>Your data is safe</p>
            </div>
        </div>
        
        <footer>
            <p><strong>SDGs Extractor v2.0</strong></p>
            <p>Deployed on Vercel • Model: facebook/bart-large-mnli</p>
        </footer>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const uploadForm = document.getElementById('uploadForm');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const submitBtn = document.getElementById('submitBtn');

        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                fileName.textContent = '📎 ' + this.files[0].name;
            }
        });

        uploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            if (!fileInput.files.length) {
                alert('Please select a PDF file');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            loading.style.display = 'block';
            results.style.display = 'none';
            submitBtn.disabled = true;

            try {
                const response = await fetch('/api/extract', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    displayResults(data);
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('Error uploading file: ' + error.message);
            } finally {
                loading.style.display = 'none';
                submitBtn.disabled = false;
            }
        });

        function displayResults(data) {
            const sdgs = data.sdg_analysis;
            let html = '<h2 style="color: #667eea; margin-bottom: 20px;">📊 Analysis Results</h2>';
            
            if (data.document && data.document.title) {
                html += `<h3 style="margin-bottom: 20px;">📄 ${data.document.title}</h3>`;
            }
            
            sdgs.forEach((sdg, index) => {
                const confidence = (sdg.confidence * 100).toFixed(1);
                let confClass = 'low';
                if (confidence >= 70) confClass = 'high';
                else if (confidence >= 40) confClass = 'medium';
                
                html += `
                    <div class="sdg-card">
                        <div class="sdg-header">
                            <h3 class="sdg-title">🎯 SDG ${sdg.sdg_number}: ${sdg.sdg_name}</h3>
                            <span class="confidence ${confClass}">${confidence}%</span>
                        </div>
                        <p class="explanation">${sdg.explanation}</p>
                        ${sdg.matched_keywords.length > 0 ? `
                        <div class="keywords">
                            <strong>🏷️ Keywords:</strong><br>
                            ${sdg.matched_keywords.map(kw => `<span class="keyword-tag">${kw}</span>`).join('')}
                        </div>
                        ` : ''}
                    </div>
                `;
            });
            
            html += '<center style="margin-top: 30px;"><button class="btn" onclick="location.reload()">📤 Analyze Another Document</button></center>';
            
            results.innerHTML = html;
            results.style.display = 'block';
            
            // Scroll to results
            results.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HOME_TEMPLATE)

@app.route('/api/extract', methods=['POST'])
def extract():
    """API endpoint for SDG extraction"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'success': False, 'error': 'Only PDF files supported'}), 400
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Extract text from PDF
            text = extract_text_from_pdf(tmp_path)
            
            if not text or len(text) < 50:
                return jsonify({
                    'success': False, 
                    'error': 'Could not extract sufficient text from PDF'
                }), 400
            
            # Clean text
            text = clean_text(text)
            
            # Extract title (first significant line)
            lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 10]
            title = lines[0][:100] if lines else "Untitled Document"
            
            # Analyze SDGs
            results = analyze_sdgs(text, top_k=3)
            
            return jsonify({
                'success': True,
                'document': {
                    'title': title,
                    'text_length': len(text)
                },
                'sdg_analysis': results
            })
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None,
        'model': 'facebook/bart-large-mnli' if classifier else 'Not loaded'
    })

# Vercel serverless function handler
def handler(environ, start_response):
    return app(environ, start_response)
