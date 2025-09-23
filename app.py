import streamlit as st
import os
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from embeddings import EmbeddingGenerator
from vector_store import PineconeVectorStore
from llm_client import GeminiLLMClient
from agents.query_understanding import QueryUnderstandingAgent
from agents.reasoning_planner import ReasoningPlannerAgent
from agents.retriever import RetrieverAgent
import uuid
from rapidfuzz import fuzz

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
    if 'documents' not in st.session_state:
        st.session_state.documents = []

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
    query_agent = QueryUnderstandingAgent(gemini_key, model_name="gemini-1.5-flash")
    retriever_agent = RetrieverAgent(embedding_generator, vector_store)
    reasoning_agent = ReasoningPlannerAgent(gemini_key, model_name="gemini-1.5-flash", retriever_agent=retriever_agent)
    
    # Test connections
    connections_ok = True
    
    # Initialize Pinecone
    if not vector_store.initialize_pinecone():
        connections_ok = False
    
    # Initialize Gemini
    if not llm_client.initialize_gemini():
        connections_ok = False
    
    return {
        'pdf_processor': pdf_processor,
        'embedding_generator': embedding_generator,
        'vector_store': vector_store,
        'llm_client': llm_client,
        'query_agent': query_agent,
        'reasoning_agent': reasoning_agent,
        'retriever_agent': retriever_agent,
        'connections_ok': connections_ok
    }

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
        components = initialize_components(pinecone_key, pinecone_env, gemini_key)
        pdf_processor = components['pdf_processor']
        embedding_generator = components['embedding_generator']
        vector_store = components['vector_store']
        llm_client = components['llm_client']
        query_agent = components['query_agent']
        reasoning_agent = components['reasoning_agent']
        retriever_agent = components['retriever_agent']
        connections_ok = components['connections_ok']
    
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
                        document_name = getattr(uploaded_file, 'name', 'Uploaded Document')
                        # Ensure index exists JIT (deferred from startup for faster load)
                        embedding_dim = embedding_generator.get_embedding_dimension()
                        vector_store.create_index(embedding_dim)
                        success = vector_store.upsert_embeddings(embedded_chunks, document_id, document_name=document_name)
                        
                        if success:
                            st.session_state.pdf_processed = True
                            st.session_state.embeddings_uploaded = True
                            st.session_state.current_document_id = document_id
                            st.rerun()

        # Document selector and management
        if st.session_state.get('documents'):
            st.markdown('<h2 class="section-header">📂 Documents</h2>', unsafe_allow_html=True)
            doc_labels = [f"{d['name']} ({d['id'][:8]}...)" for d in st.session_state.documents]
            # Determine current selection index
            current_idx = 0
            for i, d in enumerate(st.session_state.documents):
                if d['id'] == st.session_state.get('current_document_id'):
                    current_idx = i
                    break
            selected_idx = st.selectbox("Select document", list(range(len(doc_labels))), format_func=lambda i: doc_labels[i], index=current_idx)
            selected_doc = st.session_state.documents[selected_idx]
            st.session_state.current_document_id = selected_doc['id']

            # Delete document button
            if st.button("Delete Selected Document", type="secondary"):
                if vector_store.delete_document(selected_doc['id']):
                    # Remove from session list
                    st.session_state.documents = [d for d in st.session_state.documents if d['id'] != selected_doc['id']]
                    # Reset current selection
                    if st.session_state.documents:
                        st.session_state.current_document_id = st.session_state.documents[0]['id']
                    else:
                        st.session_state.current_document_id = None
                        st.session_state.pdf_processed = False
                    st.success("Document deleted.")
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
        force_reasoning = st.toggle("Force reasoning mode", value=False, help="For testing: always trigger reasoning chain display.")
        
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
                    status_box = st.status("Processing query...", state="running", expanded=True)
                    try:
                        # Step 1: Analyzing query
                        status_box.update(label="Analyzing query…", state="running")
                        qa_analysis = query_agent.analyze_query(query)
                        with st.expander("🧭 Query understanding"):
                            st.write(qa_analysis)

                        # Decide which questions to retrieve on
                        questions_for_retrieval = qa_analysis.get("sub_questions") or [query]

                        # Step 2: Let planner execute retrieval
                        status_box.update(label="Planning & retrieving…", state="running")
                        # Ensure index exists/connected in case this is a fresh session
                        try:
                            embedding_dim = embedding_generator.get_embedding_dimension()
                            vector_store.create_index(embedding_dim)
                        except Exception:
                            pass
                        reasoning_context = {
                            "query": query,
                            "sub_questions": questions_for_retrieval,
                            "intent": qa_analysis.get("intent", "unknown"),
                            "document_id": st.session_state.current_document_id,
                            "top_k": search_k,
                            "force_reasoning": force_reasoning,
                        }
                        reasoning_result = reasoning_agent.process_query(query, reasoning_context)
                        top_context = reasoning_result.get("context", {}).get("retrieved_chunks", [])
                        if not top_context:
                            status_box.update(label="No relevant context found.", state="error")
                            st.warning("Could not find relevant passages for your question.")
                            # Record explicit chat entry indicating no context
                            chat_entry = {
                                "query": query,
                                "answer": "No relevant context was found in the uploaded document for your question.",
                                "sources": [],
                                "reasoning_required": False,
                                "reasoning_chain": []
                            }
                            st.session_state.chat_history.append(chat_entry)
                            st.rerun()
                            return

                        with st.expander("🔎 Retrieval diagnostics (top matches)"):
                            for idx, it in enumerate(top_context, start=1):
                                st.markdown(
                                    f"**{idx}. Chunk {it.get('chunk_number', '?')}** — "
                                    f"vec: {it.get('score', 0.0):.3f}, "
                                    f"lex: {it.get('lex_score', 0.0):.3f}, "
                                    f"final: {it.get('final_score', 0.0):.3f}\n"
                                    f"Matched query: {it.get('base_query', query)}"
                                )
                                st.write((it.get('text', '')[:300] + '...') if len(it.get('text', '')) > 300 else it.get('text', ''))
                        
                        # Step 4: Generate final answer with reasoning
                        status_box.update(label="Generating answer…", state="running")
                        if reasoning_result.get("reasoning_required", False):
                            # Use the reasoning chain if available
                            answer = reasoning_result.get("final_answer", "")
                            reasoning_steps = reasoning_result.get("reasoning_chain", [])
                            if reasoning_steps:
                                answer += "\n\n**Reasoning Steps:**\n"
                                for i, step in enumerate(reasoning_steps, 1):
                                    answer += f"{i}. {step}\n"
                            # Annotate with context count
                            header = f"(Relevant context found: {len(top_context)} chunks)\n\n"
                            result = {"answer": header + answer, "sources": [chunk.get('source', '') for chunk in top_context]}
                        else:
                            # Fall back to simple QA for straightforward questions
                            base = llm_client.answer_question(query, top_context)
                            header = f"(Relevant context found: {len(top_context)} chunks)\n\n"
                            result = {"answer": header + base.get("answer", ""), "sources": base.get("sources", [])}

                        # Done
                        status_box.update(label="Done ✅", state="complete")

                        # Add to chat history with reasoning information
                        chat_entry = {
                            "query": query,
                            "answer": result["answer"],
                            "sources": result["sources"],
                            "reasoning_required": reasoning_result.get("reasoning_required", False),
                            "reasoning_chain": reasoning_result.get("reasoning_chain", [])
                        }
                        st.session_state.chat_history.append(chat_entry)
                        st.rerun()
                    except Exception as e:
                        status_box.update(label=f"Error: {e}", state="error")
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown('<h3 class="section-header">💭 Conversation History</h3>', unsafe_allow_html=True)
                
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    with st.expander(f"Q: {chat['query'][:100]}...", expanded=(i == 0)):
                        st.markdown(f"**Question:** {chat['query']}")
                        
                        # Show reasoning chain in a separate expander
                        if chat.get('reasoning_required') and chat.get('reasoning_chain'):
                            with st.expander("💬 View Reasoning Process", expanded=True):
                                st.markdown("### 💬 Multi-step Reasoning")
                                st.markdown("The system used the following reasoning steps to answer your question:")
                                for j, step in enumerate(chat['reasoning_chain'], 1):
                                    st.markdown(f"**Step {j}:** {step}")
                                st.markdown("---")
                        
                        # Show the final answer in a clean format
                        st.markdown("### 💬 Final Answer")
                        st.markdown(chat['answer'])
                        
                        # Show sources if available
                        if chat.get('sources'):
                            with st.expander("📝 View Sources", expanded=False):
                                st.markdown("**Sources used in this response:**")
                                for source in chat['sources']:
                                    if source:  # Only show non-empty sources
                                        st.markdown(f"- {source}")
    
    with col2:
        st.markdown('<h2 class="section-header">📊 Document Info</h2>', unsafe_allow_html=True)
        
        if st.session_state.pdf_processed:
            # Show document statistics
            # Lightweight cache for index stats to avoid frequent network calls
            def get_index_stats_cached():
                import time
                now = time.time()
                cache_key = "_cached_index_stats"
                ts_key = "_cached_index_stats_ts"
                ttl = 5.0  # seconds
                if cache_key in st.session_state and ts_key in st.session_state:
                    if now - st.session_state[ts_key] < ttl:
                        return st.session_state[cache_key]
                stats_local = vector_store.get_index_stats()
                st.session_state[cache_key] = stats_local
                st.session_state[ts_key] = now
                return stats_local

            stats = get_index_stats_cached()
            
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
