import fitz  # PyMuPDF
import re
from typing import Dict, List, Optional

class PDFExtractor:
    """Extracts Title, Abstract, and Keywords from PDF documents using PyMuPDF."""
    
    def __init__(self):
        self.title_patterns = [
            r'^title[:\s]*', 
            r'^judul[:\s]*',
            r'^document title[:\s]*',
            r'^paper title[:\s]*'
        ]
        
        self.abstract_patterns = [
            r'^abstract[:\s]*',
            r'^ringkasan[:\s]*',
            r'^intisari[:\s]*',
            r'^executive summary[:\s]*'
        ]
        
        self.keywords_patterns = [
            r'^keywords?[:\s]*',
            r'^kata kunci[:\s]*',
            r'^index terms?[:\s]*',
            r'^key terms?[:\s]*'
        ]
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, str]:
        """Extract basic metadata from PDF."""
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        
        result = {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'creation_date': metadata.get('creationDate', ''),
            'page_count': doc.page_count
        }
        
        if not result['title']:
            # Try to get title from first page
            first_page = doc[0]
            text = first_page.get_text()
            lines = text.split('\n')
            if lines:
                result['title'] = lines[0].strip()
        
        doc.close()
        return result
    
    def extract_content(self, pdf_path: str) -> Dict[str, str]:
        """Extract Title, Abstract, and Keywords from PDF content."""
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page in doc:
            full_text += page.get_text() + "\n"
        
        doc.close()
        
        # Clean and normalize text
        full_text = self._clean_text(full_text)
        
        # Extract sections
        title = self._extract_title(full_text)
        abstract = self._extract_abstract(full_text)
        keywords = self._extract_keywords(full_text)
        
        return {
            'title': title,
            'abstract': abstract,
            'keywords': keywords,
            'full_text': full_text[:5000]  # Limit for processing
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
        text = text.strip()
        return text
    
    def _extract_title(self, text: str) -> str:
        """Extract title using pattern matching."""
        lines = text.split('\n')
        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            line_lower = line.lower()
            if any(pattern in line_lower for pattern in self.title_patterns):
                if i + 1 < len(lines):
                    return lines[i + 1].strip()
            elif len(line) > 10 and len(line) < 100 and i < 5:  # Heuristic for title
                return line.strip()
        return "Untitled Document"
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract using pattern matching."""
        abstract_section = ""
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(pattern in line_lower for pattern in self.abstract_patterns):
                # Get next few lines as abstract
                abstract_lines = []
                for j in range(i + 1, min(i + 20, len(lines))):
                    if len(lines[j].strip()) > 20:  # Skip very short lines
                        abstract_lines.append(lines[j].strip())
                    if len(' '.join(abstract_lines)) > 500:  # Limit length
                        break
                abstract_section = ' '.join(abstract_lines)
                break
        
        return abstract_section[:1000] if abstract_section else text[:1000]  # Fallback
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using pattern matching."""
        keywords = []
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(pattern in line_lower for pattern in self.keywords_patterns):
                # Extract keywords after colon or pattern
                clean_line = re.sub(r'^.*?:', '', line, flags=re.IGNORECASE)
                clean_line = re.sub(r'[^\w\s,;]', '', clean_line)
                keywords_list = [k.strip() for k in re.split(r'[,;]', clean_line) if k.strip()]
                keywords.extend(keywords_list[:10])  # Limit to 10 keywords
                break
        
        return keywords[:10]  # Return maximum 10 keywords