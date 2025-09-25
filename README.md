### Planner-Driven Retrieval (LangGraph)

- The `ReasoningPlannerAgent` executes retrieval via the `RetrieverAgent` using the sub-questions or its own retrieval plan
- Retrieved chunks are stored in the planner's context and displayed in the UI diagnostics
- A "Force reasoning mode" toggle in Settings can be enabled to consistently show reasoning chains during testing

# 📚 RAG Document Assistant

A powerful Retrieval Augmented Generation (RAG) system that allows users to upload PDF documents and ask intelligent questions about their content using state-of-the-art AI technologies.

## 🌟 Features

- **📄 PDF Processing**: Upload and automatically extract text from PDF documents
- **🔪 Smart Chunking**: Intelligent document segmentation with overlap for better context
- **📊 Progress Tracking**: Real-time progress bars and status steps
  - Visual indicators during Q&A: "Analyzing query…" → "Retrieving context…" → "Generating answer…"
- **🧠 Semantic Search**: Hugging Face embeddings for accurate content retrieval
- **🗄️ Vector Storage**: Pinecone vector database for efficient similarity search
- **🤖 AI Responses**: Google Gemini 1.5 Flash API for fast, cost-effective answer generation
- **🧭 Query Understanding Agent (LangGraph)**: Classifies intent (Explain/Quiz/Summary) and optionally decomposes complex questions into sub-questions
- **📥 Retriever Agent**: Dedicated agent that performs batched sub-question embeddings, vector search, de-duplication, and hybrid re-ranking
- **🧠 Planner executes retrieval (LangGraph)**: The Reasoning/Planner Agent now calls the Retriever Agent internally and reasons over the retrieved context
- **🧪 Hybrid Re-ranking**: Combines vector similarity with lexical matching (RapidFuzz) for better retrieval precision
- **🧷 Force reasoning toggle**: UI toggle to force multi-step reasoning for testing
- **🧾 Explicit context reporting**: Every answer annotates whether relevant context was found; if not, the chat records "No relevant context was found"
- **📂 Document Management**: Sidebar document selector and one-click delete of a selected document
- **💬 Interactive UI**: Beautiful Streamlit web interface with chat history
- **📈 Analytics**: Document statistics and source attribution
- **🎤 Teaching Mode Agent (LangGraph)**: Chooses how to present responses: Explain, Quiz (with answer key), or Summary (5 lines)
- **🔄 Feedback & Reflection Agent (LangGraph)**: Evaluates answer quality and suggests a follow-up question
- **🧰 Teaching mode override**: Settings control to force presentation (Auto/Explain/Quiz/Summary)

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher** ([Download Python](https://www.python.org/downloads/))
- **pip** (Python package installer - comes with Python)
- **Git** (optional, for cloning repositories)

### Check Your Python Version:
```bash
python --version
# Should show Python 3.8.x or higher
```

## 🚀 Installation & Setup

### Step 1: Download/Clone the Project

**Option A: Download ZIP**
1. Download the project files to a folder
2. Extract all files to `C:\your-project-folder\`

**Option B: Clone with Git**
```bash
git clone <repository-url>
cd rag-document-assistant
```

### Step 2: Install Required Packages

**Method 1: Automated Installation (Recommended)**
```bash
# Run the setup script
python setup.py
```

**Method 2: Manual Installation**

**Install core packages first:**
```bash
pip install streamlit python-dotenv PyPDF2
```

**Install AI/ML packages:**
```bash
pip install google-generativeai sentence-transformers
```

**Install Pinecone (new version):**
```bash
# Remove old version if exists
pip uninstall pinecone-client -y
# Install new version
pip install pinecone
```

**Install remaining packages:**
```bash
pip install numpy pandas
```

**Install new agent & re-ranking dependencies:**
```bash
pip install langgraph langchain-core rapidfuzz
```

**Method 3: Install from Requirements File**
```bash
# Option 1: Full requirements
pip install -r requirements.txt

# Option 2: Minimal requirements (if full fails)
pip install -r requirements_minimal.txt
```

### Step 3: Get API Keys

#### 🔑 Pinecone API Key
1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Sign up/Login to your account
3. Create a new project or use existing
4. Go to "API Keys" section
5. Copy your API key and environment name

#### 🔑 Google Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

### Step 4: Configure Environment Variables

1. **Copy the example file:**
   ```bash
   copy .env.example .env
   ```

2. **Edit the `.env` file** with your API keys:
   ```env
   # Your actual API keys (replace with real values)
   PINECONE_API_KEY=your_actual_pinecone_api_key_here
   PINECONE_ENVIRONMENT=us-east-1
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

### 🔐 **Important API Key Notes:**

#### **For Testing/Demo Purposes:**
If you want to quickly test this system, you can temporarily use these keys:
- **Pinecone API**: `pcsk_7VwzkF_8T6vbouoRE2ZDinzi7LUituZ1DDQWe1GT3ehngqqSfbhzV5tLsEK4SsYdFJrLy3`
- **Pinecone Environment**: `us-east-1`
- **Gemini API**: `AIzaSyDSyRuUUxYTfc6-_7nXcFiAtGtShqckRQ8`

#### **⚠️ API Key Usage Guidelines:**
1. **Replace Gemini API key ONLY** if you encounter rate limit errors
2. **Use your own Pinecone keys** if you want to see indexing in your own account
3. **These are demo keys** - get your own for production use
4. **Never commit API keys** to public repositories (`.env` is in `.gitignore`)

#### **Rate Limit Handling:**
- The system has **automatic retry logic** for rate limits
- **Gemini 1.5 Flash** has higher rate limits than Pro
- Only replace the Gemini key if you see persistent rate limit errors

### Step 5: Test Your Setup

**Run the test script:**
```bash
python test_setup.py
```

**Expected output:**
```
🧪 Testing RAG Document Assistant Setup
==================================================
Testing package imports...
✅ Streamlit imported successfully
✅ Pinecone imported successfully
✅ Google GenerativeAI imported successfully
✅ Sentence Transformers imported successfully
✅ PyPDF2 imported successfully

Testing environment configuration...
✅ PINECONE_API_KEY is set
✅ PINECONE_ENVIRONMENT is set
✅ GEMINI_API_KEY is set

Testing custom modules...
✅ PDFProcessor imported successfully
✅ EmbeddingGenerator imported successfully
✅ PineconeVectorStore imported successfully
✅ GeminiLLMClient imported successfully

==================================================
✅ All tests passed! Your RAG system is ready to use.
```

## 🏃‍♂️ Running the Application

### Start the Application:
```bash
streamlit run app.py
```

### Access the Application:
- **Local URL**: http://localhost:8501
- **Network URL**: http://your-ip:8501

The application will automatically open in your default web browser.

## 📖 Usage Guide

### 1. Document Upload
1. **Open the application** in your browser (http://localhost:8501)
2. **In the sidebar**, click "Choose a PDF file"
3. **Select your PDF** document from your computer
4. **Click "Process Document"** to start processing

### 2. Document Processing (Automatic)
Watch real-time progress as the system:
- **Extracts text** from each PDF page (with progress bar)
- **Creates intelligent chunks** with overlap (shows chunk count)
- **Generates semantic embeddings** using Hugging Face models
- **Uploads to Pinecone** vector database

**Example Progress:**
```
📄 Starting PDF processing...
Extracting text from page 5/10
🔪 Chunking document...
Creating chunk 12/15
🧠 Generating embeddings for document chunks...
Generating embeddings: 15/15
📤 Uploading embeddings to Pinecone...
✅ Successfully uploaded 15 embeddings to Pinecone
```

### 3. Ask Questions
1. **Type your question** in the "Ask a question about your document" field
2. **Click "Get Answer"** button
3. **Watch status steps** in the app:
   - Analyzing query… (Query Understanding Agent via LangGraph)
   - Retrieving context… (Hybrid retrieval + re-ranking)
   - Generating answer… (Gemini 1.5 Flash)
4. **Get the answer** with cited chunks and diagnostics

> Note: The answer is presented via the Teaching Mode Agent. By default, the mode is selected automatically from intent, but you can force it in Settings → "Teaching mode presentation" (Auto/Explain/Quiz/Summary).

**Example Questions:**
- "What is the main topic of this document?"
- "Summarize the key findings"
- "What are the conclusions mentioned?"
- "Explain the methodology used"

### 4. Review Results
- **View AI-generated answers** in the main area
- **Check source attribution** - see which document chunks were used
- **Browse conversation history** - all Q&A pairs are saved
- **Adjust settings** - change number of chunks retrieved (1-10)

### 5. Multiple Documents
- **Upload new PDFs** - each gets a unique document ID
- **All stored in same database** - but logically separated
- **Search across all** or **filter by document**

## 🔧 Workflow Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Upload    │───▶│  Text Extraction │───▶│   Chunking      │
│   (Streamlit)   │    │    (PyPDF2)      │    │  (Intelligent)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  User Question  │───▶│   Embeddings     │───▶│   Vector DB     │
│                 │    │ (Hugging Face)   │    │  (Pinecone)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                               │
         │              ┌──────────────────┐            │
         └─────────────▶│   LLM Response   │◀───────────┘
                        │ (Gemini 1.5 Flash)│
                        └──────────────────┘

### Query Understanding Agent (LangGraph)

- Classifies intent: `explain`, `quiz`, `summary`, `other`
- Optionally decomposes complex questions into 2–5 sub-questions
- We retrieve per sub-question, merge results, and re-rank before answering

### Hybrid Re-ranking

- Vector score (Pinecone cosine similarity) blended with lexical score (RapidFuzz token set ratio)
- Default weights: `final = 0.7 * vector + 0.3 * lexical`
- See "🔎 Retrieval diagnostics" expander for per-chunk scores
```

## 🛠️ Troubleshooting

### Common Installation Issues

#### 1. **Python Version Error**
```bash
# Check Python version
python --version
# If < 3.8, download newer Python from python.org
```

#### 2. **Package Installation Fails**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install packages one by one
pip install streamlit
pip install python-dotenv
pip install PyPDF2
pip install google-generativeai
pip install sentence-transformers
pip install pinecone
```

#### 3. **Pinecone Import Error**
```bash
# Remove old version
pip uninstall pinecone-client -y
# Install new version
pip install pinecone
```

#### 4. **API Key Issues**
- **Check .env file exists** in project folder
- **Verify API keys** are correct (no extra spaces)
- **Test API keys** by running `python test_setup.py`

### Runtime Issues

#### 1. **Rate Limiting (Gemini API)**
```
Rate limit reached. Waiting 2 seconds before retry...
```
**What it means:** Gemini has throttled or your quota is temporarily exhausted.

**What we do:**
- The Query Understanding Agent and Answer generation both use automatic retry with exponential backoff.
- If the agent still cannot proceed, it falls back to a default intent (explain) with no decomposition and continues.

**Solutions:**
- Wait a few seconds and try again
- Use Gemini 1.5 Flash (already set) instead of Pro
- Check quotas in Google AI Studio and upgrade if needed
- Optionally generate a new API key under a different project/account and update `.env`

#### 2. **Chunking Issues**
```
Creating chunk 1000000/2
```
**Solution:** Already fixed in latest version - restart application

#### 3. **Streamlit Not Found**
```bash
# Install Streamlit
pip install streamlit
# Or reinstall
pip uninstall streamlit
pip install streamlit
```

#### 4. **Connection Errors**
- **Check internet connection**
- **Verify API keys are valid**
- **Check Pinecone service status**

## 🎯 Performance Tips

### 1. **Optimize Chunk Settings**
- **Smaller chunks** (500-800 chars): Better for specific questions
- **Larger chunks** (1000-1500 chars): Better for context understanding
- **Adjust in code**: Modify `chunk_size` in `PDFProcessor`

### 2. **Retrieval Settings**
- **More chunks** (7-10): More comprehensive answers
- **Fewer chunks** (3-5): Faster, more focused answers
- **Adjust in UI**: Use the slider in settings

### 3. **Model Selection**
- **Gemini 1.5 Flash**: Fast, cost-effective (current default)
- **Gemini Pro**: Higher quality, slower, more expensive
- **Change in code**: Modify `model_name` in `llm_client.py`

## 📁 Project Structure

```
rag-document-assistant/
├── 📄 app.py                    # Main Streamlit application
├── 📄 pdf_processor.py          # PDF text extraction and chunking
├── 📄 embeddings.py             # Hugging Face embedding generation
├── 📄 vector_store.py           # Pinecone vector database interface
├── 📄 llm_client.py            # Gemini API client
├── 📁 agents/                  # Agents package (LangGraph)
│   └── query_understanding.py  # Query Understanding Agent (intent + decomposition)
├── 📄 requirements.txt          # Python dependencies (full)
├── 📄 requirements_minimal.txt  # Python dependencies (minimal)
├── 📄 .env.example             # Environment variables template
├── 📄 .env                     # Your actual API keys (create this)
├── 📄 setup.py                 # Automated setup script
├── 📄 test_setup.py            # Setup verification script
├── 📄 run_setup.bat            # Windows batch setup
├── 📄 run_app.ps1              # PowerShell run script
└── 📄 README.md                # This comprehensive guide
```

## 🚀 Quick Commands Reference

### Setup Commands:
```bash
# 1. Install packages
python setup.py

# 2. Test setup
python test_setup.py

# 3. Run application
streamlit run app.py
```

### Troubleshooting Commands:
```bash
# Check Python version
python --version

# Upgrade pip
python -m pip install --upgrade pip

# Install specific package
pip install package_name

# Uninstall package
pip uninstall package_name

# List installed packages
pip list

# Check Streamlit version
streamlit version
```

### Windows-Specific Commands:
```cmd
# Run setup (Command Prompt)
run_setup.bat

# Run app (PowerShell)
.\run_app.ps1

# Copy environment file (Command Prompt)
copy .env.example .env

# Copy environment file (PowerShell)
Copy-Item .env.example .env
```

## 🎉 Success Indicators

When everything is working correctly, you should see:

1. **✅ All tests pass** when running `python test_setup.py`
2. **✅ Streamlit starts** without errors
3. **✅ API connections succeed** (Pinecone + Gemini)
4. **✅ PDF processing works** with progress bars
5. **✅ Questions get answered** with source attribution

## 🆘 Getting Help

If you encounter issues:

1. **Run the test script**: `python test_setup.py`
2. **Check the error messages** carefully
3. **Verify API keys** are correct in `.env` file
4. **Check internet connection**
5. **Try installing packages individually**

## 🎯 Next Steps

Once your RAG system is running:

1. **Try different PDFs** - academic papers, manuals, reports
2. **Experiment with questions** - specific vs. general queries
3. **Adjust settings** - chunk retrieval count, chunk sizes
4. **Monitor usage** - API costs and rate limits
5. **Customize** - modify prompts, add new features

Your RAG Document Assistant is now ready to help you extract insights from any PDF document! 🚀

## 📤 **Pushing to GitHub**

### **Step 1: Initialize Git Repository**
```bash
# Navigate to your project folder
cd "C:\Users\sratn\OneDrive\New folder"

# Initialize git repository
git init

# Add all files (except .env which is in .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: RAG Document Assistant with Pinecone, Gemini, and Streamlit"
```

### **Step 2: Create GitHub Repository**
1. Go to [GitHub.com](https://github.com)
2. Click "New Repository" (green button)
3. Name it: `rag-document-assistant`
4. Make it **Public** (so others can use it)
5. **Don't** initialize with README (we already have one)
6. Click "Create Repository"

### **Step 3: Connect and Push**
```bash
# Add GitHub repository as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/rag-document-assistant.git

# Push to GitHub
git push -u origin main
```

### **Step 4: Verify Upload**
- Check your GitHub repository
- Verify `.env` file is **NOT** uploaded (protected by `.gitignore`)
- Verify all other files are present

### **🔒 Security Notes:**
- ✅ **`.env` file is protected** by `.gitignore`
- ✅ **API keys are safe** and won't be uploaded
- ✅ **Users must create their own `.env`** file
- ✅ **Demo keys provided** in README for quick testing

## ⚙️ Configuration Options

### Embedding Models
- **Default**: `all-MiniLM-L6-v2` (384 dimensions, fast and efficient)
- **Alternative**: `all-mpnet-base-v2` (768 dimensions, higher quality)
- **Change in**: `embeddings.py` → modify `model_name` parameter

### Chunking Parameters
- **Chunk Size**: 1000 characters (adjustable)
- **Overlap**: 200 characters (prevents context loss)
- **Change in**: `pdf_processor.py` → modify `chunk_size` and `chunk_overlap`

### LLM Settings
- **Model**: Gemini 1.5 Flash (fast, cost-effective)
- **Alternative**: Gemini Pro (higher quality, slower)
- **Change in**: `llm_client.py` → modify `model_name` parameter

### Retrieval Settings
- **Top-K**: 5 similar chunks (configurable in UI slider)
- **Similarity Metric**: Cosine similarity
- **Change in**: UI settings or `vector_store.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly (`python test_setup.py`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Hugging Face** for excellent embedding models and transformers
- **Pinecone** for scalable vector database infrastructure
- **Google** for Gemini API and AI capabilities
- **Streamlit** for the amazing web framework
- **PyPDF2** for reliable PDF text extraction

---

**Happy Document Querying! 🎉📚**

*Built with ❤️ for making document analysis accessible to everyone*
