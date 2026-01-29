import os
import PyPDF2
import fitz  # PyMuPDF
from typing import List, Dict, Any


class DocumentProcessor:
    """
    Handles loading and processing of various document types including PDFs, text files, and code files.
    """
    
    def __init__(self):
        self.supported_formats = ['.txt', '.pdf', '.py', '.js', '.html', '.css', '.md', '.json', '.xml', '.csv']
    
    def load_document(self, file_path: str) -> str:
        """
        Load content from a document based on its file extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content from the document
        """
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == '.pdf':
            return self._extract_pdf_text(file_path)
        elif ext in ['.txt', '.py', '.js', '.html', '.css', '.md', '.json', '.xml']:
            return self._extract_text_file(file_path)
        elif ext == '.csv':
            return self._extract_csv_content(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: {self.supported_formats}")
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """
        Extract text from a PDF file using PyMuPDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        text = ""
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"Error reading PDF {file_path}: {str(e)}")
            # Fallback to PyPDF2
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        return text
    
    def _extract_text_file(self, file_path: str) -> str:
        """
        Extract text from a text-based file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            File content as string
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _extract_csv_content(self, file_path: str) -> str:
        """
        Extract content from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            CSV content as string with structure preserved
        """
        import pandas as pd
        df = pd.read_csv(file_path)
        return df.to_string()
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks for better context retention.
        
        Args:
            text: Input text to be chunked
            chunk_size: Maximum size of each chunk
            overlap: Number of overlapping characters between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Adjust if we're near the end of the text
            if end >= len(text):
                chunk = text[start:]
                
            chunks.append(chunk)
            start = end - overlap
            
            # Handle edge case where remaining text is shorter than overlap
            if start >= len(text):
                break
                
        return chunks
    
    def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple documents and return their content organized by file.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Dictionary mapping file paths to their processed content
        """
        results = {}
        
        for file_path in file_paths:
            try:
                content = self.load_document(file_path)
                chunks = self.chunk_text(content)
                results[file_path] = {
                    'content': content,
                    'chunks': chunks,
                    'num_chunks': len(chunks)
                }
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                results[file_path] = {'error': str(e)}
        
        return results


# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Example of processing a list of files
    # files = ["example.pdf", "example.txt"]
    # results = processor.process_documents(files)
    # print(results)