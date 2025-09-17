import streamlit as st
import os
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from embeddings import EmbeddingGenerator
from vector_store import PineconeVectorStore
from llm_client import GeminiLLMClient
import uuid

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #32cd32;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'embeddings_uploaded' not in st.session_state:
        st.session_state.embeddings_uploaded = False
    if 'current_document_id' not in st.session_state:
        st.session_state.current_document_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def check_api_keys():
    """Check if all required API keys are available"""
    pinecone_key = os.getenv('PINECONE_API_KEY')
    pinecone_env = os.getenv('PINECONE_ENVIRONMENT')
    gemini_key = os.getenv('GEMINI_API_KEY')
    
    missing_keys = []
    if not pinecone_key:
        missing_keys.append("PINECONE_API_KEY")
    if not pinecone_env:
        missing_keys.append("PINECONE_ENVIRONMENT")
    if not gemini_key:
        missing_keys.append("GEMINI_API_KEY")
    
    return missing_keys, pinecone_key, pinecone_env, gemini_key

def initialize_components(pinecone_key, pinecone_env, gemini_key):
    """Initialize all RAG components"""
    # Initialize components
    pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
    embedding_generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    vector_store = PineconeVectorStore(pinecone_key, pinecone_env)
    llm_client = GeminiLLMClient(gemini_key)
    
    # Test connections
    connections_ok = True
    
    # Initialize Pinecone
    if not vector_store.initialize_pinecone():
        connections_ok = False
    
    # Initialize Gemini
    if not llm_client.initialize_gemini():
        connections_ok = False
    
    # Create Pinecone index
    if connections_ok:
        embedding_dim = embedding_generator.get_embedding_dimension()
        if not vector_store.create_index(embedding_dim):
            connections_ok = False
    
    return pdf_processor, embedding_generator, vector_store, llm_client, connections_ok

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📚 RAG Document Assistant</h1>', unsafe_allow_html=True)
    st.markdown("Upload PDF documents and ask questions about their content using AI-powered retrieval and generation.")
    
    # Check API keys
    missing_keys, pinecone_key, pinecone_env, gemini_key = check_api_keys()
    
    if missing_keys:
        st.error(f"Missing API keys: {', '.join(missing_keys)}")
        st.info("Please create a `.env` file with your API keys. See `.env.example` for the required format.")
        st.stop()
    
    # Initialize components
    with st.spinner("Initializing RAG components..."):
        pdf_processor, embedding_generator, vector_store, llm_client, connections_ok = initialize_components(
            pinecone_key, pinecone_env, gemini_key
        )
    
    if not connections_ok:
        st.error("Failed to initialize one or more components. Please check your API keys and try again.")
        st.stop()
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.markdown('<h2 class="section-header">📄 Document Upload</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                # Process PDF
                chunks = pdf_processor.process_pdf(uploaded_file)
                
                if chunks:
                    # Generate embeddings
                    embedded_chunks = embedding_generator.embed_chunks(chunks)
                    
                    if embedded_chunks:
                        # Upload to Pinecone
                        document_id = str(uuid.uuid4())
                        success = vector_store.upsert_embeddings(embedded_chunks, document_id)
                        
                        if success:
                            st.session_state.pdf_processed = True
                            st.session_state.embeddings_uploaded = True
                            st.session_state.current_document_id = document_id
                            st.rerun()
        
        # Display processing status
        if st.session_state.pdf_processed:
            st.success("✅ Document processed successfully!")
            
            # Show index stats
            stats = vector_store.get_index_stats()
            if stats:
                st.info(f"Vectors in database: {stats.get('total_vector_count', 'Unknown')}")
        
        # Settings
        st.markdown('<h2 class="section-header">⚙️ Settings</h2>', unsafe_allow_html=True)
        
        search_k = st.slider("Number of relevant chunks to retrieve", 1, 10, 5)
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">💬 Ask Questions</h2>', unsafe_allow_html=True)
        
        if not st.session_state.pdf_processed:
            st.info("Please upload and process a PDF document first.")
        else:
            # Query input
            query = st.text_input(
                "Ask a question about your document:",
                placeholder="What is the main topic of this document?",
                key="query_input"
            )
            
            if st.button("Get Answer", type="primary") and query:
                if st.session_state.current_document_id:
                    # Generate query embedding
                    query_embedding = embedding_generator.generate_single_embedding(query)
                    
                    if query_embedding.size > 0:
                        # Search for similar chunks
                        with st.spinner("🔍 Searching for relevant information..."):
                            similar_chunks = vector_store.search_similar(
                                query_embedding, 
                                top_k=search_k,
                                document_id=st.session_state.current_document_id
                            )
                        
                        if similar_chunks:
                            # Generate answer
                            result = llm_client.answer_question(query, similar_chunks)
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                "query": query,
                                "answer": result["answer"],
                                "sources": result["sources"]
                            })
                            
                            st.rerun()
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown('<h3 class="section-header">💭 Conversation History</h3>', unsafe_allow_html=True)
                
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    with st.expander(f"Q: {chat['query'][:100]}...", expanded=(i == 0)):
                        st.markdown(f"**Question:** {chat['query']}")
                        st.markdown(f"**Answer:** {chat['answer']}")
                        
                        if chat.get('sources'):
                            st.markdown("**Sources:**")
                            for j, source in enumerate(chat['sources'][:3]):  # Show top 3 sources
                                st.markdown(f"- Chunk {source['chunk_number']} (Score: {source['score']:.3f})")
                                st.markdown(f"  *{source['preview']}*")
    
    with col2:
        st.markdown('<h2 class="section-header">📊 Document Info</h2>', unsafe_allow_html=True)
        
        if st.session_state.pdf_processed:
            # Show document statistics
            stats = vector_store.get_index_stats()
            
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Document Status:** ✅ Processed")
            if stats:
                st.markdown(f"**Total Chunks:** {stats.get('total_vector_count', 'Unknown')}")
            st.markdown(f"**Document ID:** `{st.session_state.current_document_id[:8]}...`")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Model information
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Embedding Model:** all-MiniLM-L6-v2")
            st.markdown("**LLM Model:** Gemini 1.5 Flash")
            st.markdown("**Vector Database:** Pinecone")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No document processed yet.")
        
        # Help section
        st.markdown('<h2 class="section-header">❓ How to Use</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        1. **Upload PDF**: Choose a PDF file from your computer
        2. **Process**: Click "Process Document" to extract and chunk text
        3. **Ask Questions**: Type questions about the document content
        4. **Get Answers**: AI will provide answers based on the document
        
        **Tips:**
        - Ask specific questions for better results
        - The system will show which parts of the document were used
        - Adjust the number of chunks to retrieve in settings
        """)

if __name__ == "__main__":
    main()
