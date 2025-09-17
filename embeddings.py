from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
import streamlit as st

class EmbeddingGenerator:
    """Handles text embeddings using Hugging Face models"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator with specified model
        
        Args:
            model_name: Hugging Face model name for embeddings
                       Default: all-MiniLM-L6-v2 (384 dimensions, good balance of speed/quality)
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = None
        
    @st.cache_resource
    def load_model(_self):
        """Load the sentence transformer model (cached for performance)"""
        try:
            with st.spinner(f"Loading embedding model: {_self.model_name}..."):
                model = SentenceTransformer(_self.model_name)
                
                # Get embedding dimension
                test_embedding = model.encode(["test"])
                embedding_dim = test_embedding.shape[1]
                
                st.success(f"✅ Loaded embedding model with {embedding_dim} dimensions")
                return model, embedding_dim
                
        except Exception as e:
            st.error(f"Error loading embedding model: {str(e)}")
            return None, None
    
    def get_model(self):
        """Get the loaded model and embedding dimension"""
        if self.model is None:
            self.model, self.embedding_dimension = self.load_model()
        return self.model, self.embedding_dimension
    
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of numpy arrays containing embeddings
        """
        model, _ = self.get_model()
        
        if model is None:
            st.error("Embedding model not loaded")
            return []
        
        try:
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            embeddings = []
            batch_size = 32  # Process in batches for better performance
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Generate embeddings for batch
                batch_embeddings = model.encode(batch, convert_to_numpy=True)
                embeddings.extend(batch_embeddings)
                
                # Update progress
                progress = min(1.0, (i + batch_size) / len(texts))
                progress_bar.progress(progress)
                status_text.text(f"Generating embeddings: {min(i + batch_size, len(texts))}/{len(texts)}")
            
            progress_bar.empty()
            status_text.empty()
            
            return embeddings
            
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return []
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string to embed
            
        Returns:
            Numpy array containing embedding
        """
        model, _ = self.get_model()
        
        if model is None:
            st.error("Embedding model not loaded")
            return np.array([])
        
        try:
            embedding = model.encode([text], convert_to_numpy=True)[0]
            return embedding
            
        except Exception as e:
            st.error(f"Error generating single embedding: {str(e)}")
            return np.array([])
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        _, dimension = self.get_model()
        return dimension if dimension else 384  # Default fallback
    
    def embed_chunks(self, chunks: List[Tuple[str, int]]) -> List[Tuple[np.ndarray, str, int]]:
        """
        Generate embeddings for document chunks
        
        Args:
            chunks: List of (text, chunk_number) tuples
            
        Returns:
            List of (embedding, text, chunk_number) tuples
        """
        if not chunks:
            return []
        
        st.info("🧠 Generating embeddings for document chunks...")
        
        # Extract texts from chunks
        texts = [chunk[0] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        if not embeddings:
            return []
        
        # Combine embeddings with original chunk data
        embedded_chunks = []
        for i, (text, chunk_num) in enumerate(chunks):
            if i < len(embeddings):
                embedded_chunks.append((embeddings[i], text, chunk_num))
        
        st.success(f"✅ Generated embeddings for {len(embedded_chunks)} chunks")
        
        return embedded_chunks
