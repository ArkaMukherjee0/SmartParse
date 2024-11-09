import streamlit as st
import PyPDF2
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_groq import ChatGroq
from rouge import Rouge
import os
from dotenv import load_dotenv

# Load environment variables and setup
load_dotenv()
key = os.getenv('GROQ_API_KEY')

# Page configuration
st.set_page_config(
    page_title="SmartParse: AI-Powered PDF Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False

# Modified get_custom_css() function with fixed text colors
def get_custom_css():
    if st.session_state.theme == 'dark':
        return """
        <style>
            :root {
                --background-color: #1E1E1E;
                --text-color: #FFFFFF;
                --card-background: #2D2D2D;
                --border-color: #404040;
                --hover-color: #3D3D3D;
                --shadow-color: rgba(0,0,0,0.3);
                --sidebar-bg: #2D2D2D;
            }
            .main {
                background-color: var(--background-color);
            }
            .chat-message {
                background-color: var(--card-background);
                border: 1px solid var(--border-color);
                margin: 1rem 0;
                padding: 1rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px var(--shadow-color);
            }
            .chat-message strong {
                color: var(--text-color);
            }
            .chat-message div {
                color: var(--text-color);
            }
            .user-message {
                background-color: #2B5C34;
                color: var(--text-color) !important;
            }
            .assistant-message {
                background-color: var(--card-background);
                color: var(--text-color) !important;
            }
            .upload-section {
                background-color: var(--card-background);
                border: 2px dashed var(--border-color);
            }
            .source-section {
                background-color: var(--card-background);
            }
            .metrics-card {
                background-color: var(--card-background);
                border: 1px solid var(--border-color);
            }
            .stApp {
                background-color: var(--background-color);
                color: var(--text-color);
            }
            .sidebar .sidebar-content {
                background-color: var(--sidebar-bg);
                color: var(--text-color);
            }
            /* Fix text color in all Streamlit elements */
            .stMarkdown, .stText, h1, h2, h3, p {
                color: var(--text-color) !important;
            }
            .sidebar-feature {
                background-color: #3D3D3D;
                padding: 0.8rem;
                margin: 0.5rem 0;
                border-radius: 8px;
                border-left: 4px solid #00CC66;
                transition: transform 0.2s;
                color: var(--text-color);
            }
            .sidebar-feature:hover {
                transform: translateX(5px);
            }
            .sidebar-feature p {
                color: var(--text-color) !important;
            }
            .disclaimer {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                text-align: center;
                padding: 1rem;
                background-color: var(--background-color);
                border-top: 1px solid var(--border-color);
                font-size: 0.9rem;
                color: #666;
            }
            /* Style for code blocks */
            code {
                color: var(--text-color) !important;
                background-color: var(--card-background) !important;
            }
            pre {
                background-color: var(--card-background) !important;
            }
            /* Fix input text color */
            .stTextInput input {
                color: var(--text-color) !important;
            }
        </style>
        """
    else:
        return """
        <style>
            :root {
                --background-color: #FFFFFF;
                --text-color: #000000;
                --card-background: #F8F9FA;
                --border-color: #DEE2E6;
                --hover-color: #F1F3F5;
                --shadow-color: rgba(0,0,0,0.1);
                --sidebar-bg: #F8F9FA;
            }
            .main {
                background-color: var(--background-color);
            }
            .chat-message {
                background-color: var(--card-background);
                border: 1px solid var(--border-color);
                margin: 1rem 0;
                padding: 1rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px var(--shadow-color);
            }
            .chat-message strong {
                color: var(--text-color);
            }
            .chat-message div {
                color: var(--text-color);
            }
            .user-message {
                background-color: #E3F2E6;
                color: var(--text-color) !important;
            }
            .assistant-message {
                background-color: var(--card-background);
                color: var(--text-color) !important;
            }
            .upload-section {
                background-color: var(--card-background);
                border: 2px dashed var(--border-color);
            }
            .source-section {
                background-color: var(--card-background);
            }
            .metrics-card {
                background-color: var(--card-background);
                border: 1px solid var(--border-color);
            }
            .stApp {
                background-color: var(--background-color);
                color: var(--text-color);
            }
            /* Fix text color in all Streamlit elements */
            .stMarkdown, .stText, h1, h2, h3, p {
                color: var(--text-color) !important;
            }
            .sidebar .sidebar-content {
                padding: 2rem 1rem;
                background-color: var(--sidebar-bg);
                color: var(--text-color);
            }
            .sidebar-feature {
                background-color: #F8F9FA;
                padding: 0.8rem;
                margin: 0.5rem 0;
                border-radius: 8px;
                border-left: 4px solid #00CC66;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                transition: transform 0.2s;
                color: var(--text-color);
            }
            .sidebar-feature:hover {
                transform: translateX(5px);
            }
            .sidebar-feature p {
                color: var(--text-color) !important;
            }
            .disclaimer {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                text-align: center;
                padding: 1rem;
                background-color: var(--background-color);
                border-top: 1px solid var(--border-color);
                font-size: 0.9rem;
                color: #666;
            }
            /* Style for code blocks */
            code {
                color: var(--text-color) !important;
                background-color: var(--card-background) !important;
            }
            pre {
                background-color: var(--card-background) !important;
            }
            /* Fix input text color */
            .stTextInput input {
                color: var(--text-color) !important;
            }
        </style>
        """

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Function to calculate ROUGE scores
def calculate_rouge_scores(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    return scores

# Initialize Groq model
llm_groq = ChatGroq(model="mixtral-8x7b-32768", groq_api_key=key)



# Enhanced Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/ArkaMukherjee0/SmartParse/main/assets/logo.png", use_column_width=True)
    
    # Theme toggle with better styling
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 🎨 Theme")
    with col2:
        theme = st.toggle("🌓", value=st.session_state.theme == 'dark')
        if theme:
            st.session_state.theme = 'dark'
        else:
            st.session_state.theme = 'light'
    
    st.markdown("### 🚀 Features")
    features = {
        "📝 Document Analysis": "Advanced PDF text extraction and processing",
        "🤖 AI Chat": "Natural conversation with your documents",
        "💾 History": "Full conversation history and tracking",
        "🎨 Themes": "Customizable dark and light modes",
        "📊 Analytics": "ROUGE score analysis and metrics"
    }
    
    for title, description in features.items():
        st.markdown(f"""
            <div class="sidebar-feature">
                <strong>{title}</strong>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">{description}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ About SmartParse"):
        st.markdown("""
            SmartParse transforms your PDF documents into interactive conversations.
            Upload your PDF and chat naturally about its contents!
        """)

# Main content area
st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1>📚 SmartParse</h1>
        <p style='font-size: 1.2em;'>Your AI-Powered PDF Companion</p>
    </div>
""", unsafe_allow_html=True)

# File upload section
uploaded_file = st.file_uploader("Drop your PDF here or click to upload", type="pdf")

if uploaded_file and not st.session_state.processing_done:
    with st.spinner("🔄 Processing your document..."):
        # PDF processing logic
        pdf = PyPDF2.PdfReader(uploaded_file)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        texts = text_splitter.split_text(pdf_text)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': "cpu"}
        )
        docsearch = FAISS.from_texts(texts, embeddings)
        
        message_history = ChatMessageHistory()
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )
        
        st.session_state.chain = ConversationalRetrievalChain.from_llm(
            llm=llm_groq,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
            memory=memory,
            return_source_documents=True,
        )
        st.session_state.pdf_text = pdf_text
        st.session_state.processing_done = True
    
    st.success("✨ Document processed successfully!")

if uploaded_file:
    # Chat interface
    st.markdown("### 💬 Chat with your PDF")
    
    # Create a container for the chat history
    chat_container = st.container()
    
    # Display chat history in the container
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>Assistant:</strong> {message["content"]}
                    </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    with st.form(key="chat_form"):
        user_input = st.text_input("Ask a question about your document:", key="user_input")
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get response
            with st.spinner("🤔 Thinking..."):
                res = st.session_state.chain.invoke(user_input)
                answer = res["answer"]
                source_documents = res.get("source_documents", [])
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                # Store source documents in session state
                st.session_state.last_sources = source_documents
            
            # Clear the input and rerun to update the chat display
            st.rerun()
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Expandable sections for sources and metrics
    with st.expander("📑 View Source References"):
        if hasattr(st.session_state, 'last_sources'):
            for idx, doc in enumerate(st.session_state.last_sources):
                st.markdown(f"**Source {idx + 1}:**")
                st.markdown(f"```\n{doc.page_content}\n```")
    
    with st.expander("📊 View Response Metrics"):
        if len(st.session_state.chat_history) > 0 and st.session_state.chat_history[-1]["role"] == "assistant":
            last_response = st.session_state.chat_history[-1]["content"]
            rouge_scores = calculate_rouge_scores(st.session_state.pdf_text, last_response)
            cols = st.columns(3)
            with cols[0]:
                st.metric("ROUGE-1", f"{rouge_scores['rouge-1']['f']:.3f}")
            with cols[1]:
                st.metric("ROUGE-2", f"{rouge_scores['rouge-2']['f']:.3f}")
            with cols[2]:
                st.metric("ROUGE-L", f"{rouge_scores['rouge-l']['f']:.3f}")

else:
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h3>👋 Welcome to SmartParse!</h3>
            <p>Upload a PDF document to start an AI-powered conversation about its contents.</p>
        </div>
    """, unsafe_allow_html=True)

# Add disclaimer at the bottom
st.markdown("""
    <div class="disclaimer">
        ⚠️ SmartParse can make mistakes. Please double-check responses.
    </div>
""", unsafe_allow_html=True)