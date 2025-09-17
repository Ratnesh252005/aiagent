import google.generativeai as genai
from typing import List, Dict, Any
import streamlit as st
import time

class GeminiLLMClient:
    """Handles Gemini API interactions for generating responses"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize Gemini LLM client
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model to use (default: gemini-1.5-flash for speed and cost efficiency)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        
    def initialize_gemini(self):
        """Initialize Gemini API connection"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            st.success("✅ Connected to Gemini API")
            return True
            
        except Exception as e:
            st.error(f"Error connecting to Gemini API: {str(e)}")
            return False
    
    def create_rag_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Create a RAG prompt with query and retrieved context
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant chunks from vector store
            
        Returns:
            Formatted prompt for the LLM
        """
        # Build context from retrieved chunks
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_text += f"Context {i} (Chunk {chunk['chunk_number']}, Score: {chunk['score']:.3f}):\n"
            context_text += f"{chunk['text']}\n\n"
        
        # Create the RAG prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided document context. 
Use the following context to answer the user's question. If the answer cannot be found in the context, 
say so clearly and don't make up information.

CONTEXT:
{context_text}

QUESTION: {query}

INSTRUCTIONS:
1. Answer based only on the provided context
2. If the context doesn't contain enough information, say "I cannot find sufficient information in the provided document to answer this question."
3. Be specific and cite relevant parts of the context when possible
4. Keep your answer concise but comprehensive
5. If multiple context chunks are relevant, synthesize information from them

ANSWER:"""
        
        return prompt
    
    def generate_response(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate response using Gemini API with retry logic
        
        Args:
            prompt: Input prompt for the model
            max_retries: Maximum number of retry attempts
            
        Returns:
            Generated response text
        """
        if not self.model:
            st.error("Gemini model not initialized")
            return "Error: Gemini model not available"
        
        for attempt in range(max_retries):
            try:
                # Generate response
                response = self.model.generate_content(prompt)
                
                if response.text:
                    return response.text.strip()
                else:
                    return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle rate limiting
                if "quota" in error_msg or "rate" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Exponential backoff
                        st.warning(f"Rate limit reached. Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return "I'm currently experiencing high demand. Please try again in a few moments."
                
                # Handle other errors
                elif attempt < max_retries - 1:
                    st.warning(f"Attempt {attempt + 1} failed. Retrying...")
                    time.sleep(1)
                    continue
                else:
                    st.error(f"Error generating response: {str(e)}")
                    return f"I encountered an error while processing your request: {str(e)}"
        
        return "I apologize, but I couldn't generate a response after multiple attempts."
    
    def answer_question(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Complete RAG pipeline: create prompt and generate answer
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant chunks
            
        Returns:
            Dictionary containing answer and metadata
        """
        if not context_chunks:
            return {
                "answer": "I couldn't find any relevant information in the document to answer your question.",
                "sources": [],
                "context_used": False
            }
        
        # Create RAG prompt
        prompt = self.create_rag_prompt(query, context_chunks)
        
        # Generate response
        with st.spinner("🤖 Generating answer..."):
            answer = self.generate_response(prompt)
        
        # Prepare source information
        sources = []
        for chunk in context_chunks:
            sources.append({
                "chunk_number": chunk["chunk_number"],
                "score": chunk["score"],
                "preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": True,
            "num_sources": len(context_chunks)
        }
    
    def test_connection(self) -> bool:
        """Test Gemini API connection with a simple query"""
        try:
            if not self.model:
                return False
            
            test_response = self.model.generate_content("Hello, please respond with 'Connection successful'")
            return test_response.text is not None
            
        except Exception as e:
            st.error(f"Gemini API test failed: {str(e)}")
            return False
