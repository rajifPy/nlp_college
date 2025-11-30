import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from supabase import create_client, Client
from dotenv import load_dotenv
from utils.pdf_extractor import PDFExtractor
from utils.rule_based_matcher import RuleBasedMatcher
from utils.semantic_matcher import SemanticMatcher
from utils.model_loader import ModelLoader
import tempfile

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key_for_development')

# Initialize Supabase client (optional for local development)
supabase_url = os.environ.get('SUPABASE_URL')
supabase_key = os.environ.get('SUPABASE_KEY')

if supabase_url and supabase_key:
    supabase: Client = create_client(supabase_url, supabase_key)
else:
    supabase = None
    print("ℹ️ SUPABASE_URL or SUPABASE_KEY not set. Database operations will be disabled.")

# Initialize extractors and matchers
pdf_extractor = PDFExtractor()
rule_matcher = RuleBasedMatcher('data/sdg_patterns.csv')
model_loader = ModelLoader(os.getenv('MODEL_PATH', './models'))
semantic_matcher = SemanticMatcher(model_loader)

@app.route('/')
def index():
    """Home page with upload form."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF file upload and analysis."""
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if not file.filename.lower().endswith('.pdf'):
        flash('Only PDF files are supported', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        
        # Extract content from PDF
        metadata = pdf_extractor.extract_metadata(file_path)
        content = pdf_extractor.extract_content(file_path)
        
        # Combine text for analysis
        analysis_text = f"{content['title']} {content['abstract']} {' '.join(content['keywords'])} {content['full_text']}"
        
        # Get predictions from your trained model
        model_results = model_loader.predict_sdgs(analysis_text, top_k=3)
        
        # Generate extraction ID
        extraction_id = str(uuid.uuid4())
        
        # Save to database if available
        if supabase:
            try:
                user_id = session.get('user_id', 'anonymous_user')
                
                response = supabase.table('extractions').insert({
                    'id': extraction_id,
                    'user_id': user_id,
                    'document_name': file.filename,
                    'title': content['title'],
                    'abstract': content['abstract'],
                    'keywords': content['keywords'],
                    'sdg_results': model_results,
                    'created_at': datetime.now().isoformat()
                }).execute()
                print("✅ Saved to database:", response)
            except Exception as e:
                print(f"⚠️ Database error: {e}")
        
        # Clean up temporary file
        os.remove(file_path)
        os.rmdir(temp_dir)
        
        # Store results in session for results page
        session['extraction_results'] = {
            'metadata': metadata,
            'content': content,
            'model_results': model_results,
            'extraction_id': extraction_id,
            'document_name': file.filename
        }
        
        return redirect(url_for('results'))
        
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    """Display analysis results."""
    if 'extraction_results' not in session:
        flash('No results found. Please upload a file first.', 'error')
        return redirect(url_for('index'))
    
    results_data = session['extraction_results']
    return render_template('results.html', **results_data)

@app.route('/history')
def history():
    """Display extraction history."""
    extractions = []
    
    if supabase:
        try:
            user_id = session.get('user_id', 'anonymous_user')
            response = supabase.table('extractions')\
                .select('*')\
                .eq('user_id', user_id)\
                .order('created_at', desc=True)\
                .limit(10)\
                .execute()
            
            if response.data:
                extractions = response.data
        except Exception as e:
            print(f"⚠️ Database error: {e}")
    
    return render_template('history.html', extractions=extractions)

@app.route('/api/extract', methods=['POST'])
def api_extract():
    """API endpoint for SDG extraction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400
    
    try:
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        
        # Extract content
        content = pdf_extractor.extract_content(file_path)
        analysis_text = f"{content['title']} {content['abstract']} {' '.join(content['keywords'])} {content['full_text']}"
        
        # Run analysis with your model
        model_results = model_loader.predict_sdgs(analysis_text, top_k=3)
        
        # Clean up
        os.remove(file_path)
        os.rmdir(temp_dir)
        
        return jsonify({
            'success': True,
            'document': {
                'title': content['title'],
                'abstract': content['abstract'],
                'keywords': content['keywords']
            },
            'sdg_analysis': model_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-model')
def test_model():
    """Test page untuk memeriksa model loading"""
    try:
        # Test dengan text sample
        test_text = "This project aims to reduce poverty in rural communities through sustainable agriculture and clean water access."
        results = model_loader.predict_sdgs(test_text, top_k=3)
        
        return render_template('test_model.html', 
                             model_loaded=model_loader.sdg_model is not None,
                             results=results,
                             test_text=test_text)
    except Exception as e:
        return f"Error testing model: {str(e)}", 500

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("🚀 Starting SDGs Extractor application...")
    print("🔧 Development server running at http://localhost:5000")
    print("📄 Upload PDF files to analyze SDG alignment")
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=os.environ.get('DEBUG', 'True') == 'True')