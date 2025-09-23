from pinecone import Pinecone, ServerlessSpec
import numpy as np
from typing import List, Tuple, Dict, Any
import streamlit as st
import os
from datetime import datetime
import uuid

class PineconeVectorStore:
    """Handles Pinecone vector database operations"""
    
    def __init__(self, api_key: str, environment: str, index_name: str = "rag-documents"):
        """
        Initialize Pinecone vector store
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (now used as cloud region)
            index_name: Name of the Pinecone index
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.pc = None
        self.index = None
        
    def initialize_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            self.pc = Pinecone(api_key=self.api_key)
            st.success("✅ Connected to Pinecone")
            return True
            
        except Exception as e:
            st.error(f"Error connecting to Pinecone: {str(e)}")
            return False

    def _ensure_index_connected(self) -> bool:
        """Ensure self.index is connected; attempt lazy connection if possible."""
        if self.index is not None:
            return True
        if self.pc is None:
            st.error("Pinecone client not initialized. Please check API key and restart.")
            return False
        try:
            # Try to connect to existing index without creating
            self.index = self.pc.Index(self.index_name)
            return True
        except Exception as e:
            st.error(f"Pinecone index not initialized or not found: {self.index_name}. Process a document first to create it. Details: {str(e)}")
            return False
    
    def create_index(self, dimension: int, metric: str = "cosine"):
        """
        Create Pinecone index if it doesn't exist
        
        Args:
            dimension: Embedding dimension
            metric: Distance metric (cosine, euclidean, dotproduct)
        """
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                st.info(f"Creating new Pinecone index: {self.index_name}")
                
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.environment
                    )
                )
                
                st.success(f"✅ Created index '{self.index_name}' with dimension {dimension}")
            else:
                st.info(f"Using existing index: {self.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            return True
            
        except Exception as e:
            st.error(f"Error creating/connecting to index: {str(e)}")
            return False
    
    def upsert_embeddings(self, embedded_chunks: List[Tuple[np.ndarray, str, int]], 
                         document_id: str = None,
                         document_name: str = None) -> bool:
        """
        Upload embeddings to Pinecone
        
        Args:
            embedded_chunks: List of (embedding, text, chunk_number) tuples
            document_id: Unique identifier for the document
            
        Returns:
            Success status
        """
        if not self._ensure_index_connected():
            return False
        
        if not embedded_chunks:
            st.warning("No embeddings to upload")
            return False
        
        try:
            # Generate document ID if not provided
            if not document_id:
                document_id = str(uuid.uuid4())
            
            st.info("📤 Uploading embeddings to Pinecone...")
            
            # Prepare vectors for upsert
            vectors = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (embedding, text, chunk_num) in enumerate(embedded_chunks):
                vector_id = f"{document_id}_chunk_{chunk_num}"
                
                # Prepare metadata
                metadata = {
                    "text": text[:1000],  # Limit text size for metadata
                    "chunk_number": chunk_num,
                    "document_id": document_id,
                    "document_name": document_name or "Untitled Document",
                    "timestamp": datetime.now().isoformat(),
                    "text_length": len(text)
                }
                
                # Convert numpy array to list for Pinecone
                embedding_list = embedding.tolist()
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding_list,
                    "metadata": metadata
                })
                
                # Update progress
                progress = (i + 1) / len(embedded_chunks)
                progress_bar.progress(progress)
                status_text.text(f"Preparing vector {i + 1}/{len(embedded_chunks)}")
            
            # Upsert vectors in batches
            batch_size = 100
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                
                # Update progress
                batch_num = (i // batch_size) + 1
                progress = batch_num / total_batches
                progress_bar.progress(progress)
                status_text.text(f"Uploading batch {batch_num}/{total_batches}")
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Successfully uploaded {len(vectors)} embeddings to Pinecone")
            
            # Store document registry in session state for later use
            if 'documents' not in st.session_state:
                st.session_state.documents = []
            # Avoid duplicates
            if not any(d.get('id') == document_id for d in st.session_state.documents):
                st.session_state.documents.append({
                    'id': document_id,
                    'name': document_name or 'Untitled Document',
                    'vector_count': len(vectors),
                    'created_at': datetime.now().isoformat()
                })
            st.session_state.current_document_id = document_id
            
            return True
            
        except Exception as e:
            st.error(f"Error uploading embeddings: {str(e)}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5, 
                      document_id: str = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Pinecone
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of similar results to return
            document_id: Optional document ID to filter results
            
        Returns:
            List of similar chunks with metadata
        """
        if not self._ensure_index_connected():
            return []
        
        try:
            # Prepare query
            query_vector = query_embedding.tolist()
            
            # Add filter if document_id is specified
            filter_dict = None
            if document_id:
                filter_dict = {"document_id": {"$eq": document_id}}
            
            # Perform search
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Process results
            similar_chunks = []
            for match in results.matches:
                chunk_data = {
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "chunk_number": match.metadata.get("chunk_number", 0),
                    "document_id": match.metadata.get("document_id", ""),
                    "timestamp": match.metadata.get("timestamp", "")
                }
                similar_chunks.append(chunk_data)
            
            return similar_chunks
            
        except Exception as e:
            st.error(f"Error searching vectors: {str(e)}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index"""
        if not self._ensure_index_connected():
            return {}
        
        try:
            stats = self.index.describe_index_stats()
            return stats
            
        except Exception as e:
            st.error(f"Error getting index stats: {str(e)}")
            return {}
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all vectors for a specific document
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Success status
        """
        if not self.index:
            st.error("Pinecone index not initialized")
            return False
        
        try:
            # Delete vectors with matching document_id
            self.index.delete(filter={"document_id": {"$eq": document_id}})
            st.success(f"✅ Deleted document {document_id} from vector store")
            return True
            
        except Exception as e:
            st.error(f"Error deleting document: {str(e)}")
            return False
