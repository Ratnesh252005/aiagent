import PyPDF2
import re
from typing import List, Tuple
import streamlit as st

class PDFProcessor:
    """Handles PDF text extraction and chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            # Create progress bar for PDF extraction
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_pages = len(pdf_reader.pages)
            
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += page_text + "\n"
                
                # Update progress
                progress = (i + 1) / total_pages
                progress_bar.progress(progress)
                status_text.text(f"Extracting text from page {i + 1}/{total_pages}")
            
            progress_bar.empty()
            status_text.empty()
            
            return self.clean_text(text)
        
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/]', '', text)
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[Tuple[str, int]]:
        """Split text into overlapping chunks with chunk numbers"""
        if not text:
            return []
        
        chunks = []
        words = text.split()
        
        # Calculate approximate words per chunk
        words_per_chunk = max(50, self.chunk_size // 5)  # Minimum 50 words per chunk
        overlap_words = min(words_per_chunk // 4, self.chunk_overlap // 5)  # Max 25% overlap
        
        # Ensure we have a reasonable step size
        step_size = max(10, words_per_chunk - overlap_words)  # Minimum step of 10 words
        
        # Calculate total chunks more accurately
        total_chunks = max(1, (len(words) - overlap_words) // step_size + 1)
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_idx = 0
        chunk_num = 0
        
        while start_idx < len(words):
            # Calculate end index for this chunk
            end_idx = min(start_idx + words_per_chunk, len(words))
            
            # Extract chunk text
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            # Only add non-empty chunks
            if chunk_text.strip():
                chunks.append((chunk_text.strip(), chunk_num))
                chunk_num += 1
            
            # Update progress
            progress = min(1.0, start_idx / len(words))
            progress_bar.progress(progress)
            status_text.text(f"Creating chunk {chunk_num}/{total_chunks}")
            
            # Move start index for next chunk (with proper step size)
            start_idx += step_size
            
            # Safety check to prevent infinite loops
            if start_idx <= (end_idx - step_size) and chunk_num > 1000:
                st.error("Chunking stopped due to potential infinite loop")
                break
        
        progress_bar.empty()
        status_text.empty()
        
        return chunks
    
    def process_pdf(self, pdf_file) -> List[Tuple[str, int]]:
        """Complete PDF processing pipeline"""
        st.info("📄 Starting PDF processing...")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_file)
        
        if not text:
            st.error("No text could be extracted from the PDF")
            return []
        
        st.success(f"✅ Extracted {len(text)} characters from PDF")
        
        # Chunk text
        st.info("🔪 Chunking document...")
        chunks = self.chunk_text(text)
        
        if chunks:
            st.success(f"✅ Created {len(chunks)} chunks from document")
            
            # Display chunking summary
            with st.expander("📊 Chunking Summary"):
                st.write(f"**Total chunks created:** {len(chunks)}")
                st.write(f"**Chunk size:** ~{self.chunk_size} characters")
                st.write(f"**Chunk overlap:** ~{self.chunk_overlap} characters")
                
                # Show first few chunks as preview
                st.write("**Preview of first 3 chunks:**")
                for i, (chunk_text, chunk_num) in enumerate(chunks[:3]):
                    st.write(f"**Chunk {chunk_num}:** {chunk_text[:200]}...")
        
        return chunks
