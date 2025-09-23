from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Optional
import os
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
        # Performance knobs via env
        self.device = os.getenv("EMBEDDING_DEVICE", "cpu")  # e.g., "cuda" if available
        try:
            self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
        except Exception:
            self.batch_size = 64
        
    @st.cache_resource
    def load_model(_self):
        """Load the sentence transformer model (cached for performance)"""
        try:
            with st.spinner(f"Loading embedding model: {_self.model_name} ({_self.device})..."):
                model = SentenceTransformer(_self.model_name, device=_self.device)
                # Use built-in method to get dimension (avoids extra forward pass)
                try:
                    embedding_dim = int(getattr(model, "get_sentence_embedding_dimension", lambda: 0)())
                    if not embedding_dim:
                        # Fallback (single quick encode) if method unavailable
                        test_embedding = model.encode(["t"], convert_to_numpy=True)
                        embedding_dim = test_embedding.shape[1]
                except Exception:
                    test_embedding = model.encode(["t"], convert_to_numpy=True)
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
            # Fast path for small N: no UI, single call
            if len(texts) <= 8:
                arr = model.encode(texts, convert_to_numpy=True, batch_size=self.batch_size)
                return list(arr)

            # UI path for larger batches
            progress_bar = st.progress(0)
            status_text = st.empty()

            embeddings: List[np.ndarray] = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = model.encode(batch, convert_to_numpy=True, batch_size=self.batch_size)
                embeddings.extend(batch_embeddings)
                progress = min(1.0, (i + self.batch_size) / len(texts))
                progress_bar.progress(progress)
                status_text.text(f"Generating embeddings: {min(i + self.batch_size, len(texts))}/{len(texts)}")

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
            embedding = model.encode([text], convert_to_numpy=True, batch_size=1)[0]
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

    # --- Performance: quiet batch embedding without Streamlit UI (for agents) ---
    def generate_embeddings_quiet(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embedding without Streamlit progress to minimize overhead for small agent calls."""
        if not texts:
            return []
        model, _ = self.get_model()
        if model is None:
            return []
        try:
            arr = model.encode(texts, convert_to_numpy=True, batch_size=self.batch_size)
            return list(arr)
        except Exception:
            return []
