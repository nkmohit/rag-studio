import streamlit as st
import requests
from datetime import datetime

# Backend API URL
BACKEND_URL = "http://localhost:8000"

# Page config
st.set_page_config(page_title="RAG Studio", page_icon="📚", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        margin-bottom: 2rem;
    }
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        background-color: #4CAF50;
        color: white;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin: 0.25rem;
    }
    [data-testid="column"]:first-child {
        border-right: none;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []

# Helper function to fetch documents
def fetch_documents():
    try:
        response = requests.get(f"{BACKEND_URL}/documents")
        response.raise_for_status()
        return response.json().get("documents", [])
    except Exception as e:
        st.error(f"Failed to fetch documents: {str(e)}")
        return []

# Helper function to delete document
def delete_document(filename):
    try:
        response = requests.delete(f"{BACKEND_URL}/documents/{filename}")
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Failed to delete document: {str(e)}")
        return False

# Header
st.markdown('<div class="main-header">📚 RAG Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Chat with your documents using Retrieval-Augmented Generation</div>', unsafe_allow_html=True)

# Create two columns: left for documents, right for chat
left_col, right_col = st.columns([1, 2], gap="large")

# LEFT PANEL: Document Management
with left_col:
    st.subheader("📂 Document Library")
    
    # Upload section
    with st.expander("➕ Upload New Document", expanded=True):
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=["pdf", "txt"],
            help="Upload a document to add to your knowledge base",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📤 Upload & Index", use_container_width=True):
                    with st.spinner("Processing..."):
                        try:
                            # Index file
                            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                            index_response = requests.post(f"{BACKEND_URL}/index", files=files)
                            index_response.raise_for_status()
                            index_data = index_response.json()
                            
                            if index_data.get("status") == "indexed":
                                st.success(f"✓ {uploaded_file.name} indexed!")
                                st.session_state.documents = fetch_documents()
                                st.rerun()
                            else:
                                st.error(f"Indexing failed: {index_data.get('message', 'Unknown error')}")
                                
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            with col2:
                if st.button("✖ Cancel", use_container_width=True):
                    st.rerun()
    
    st.divider()
    
    # Fetch and display documents
    st.session_state.documents = fetch_documents()
    
    if st.session_state.documents:
        st.markdown(f"**{len(st.session_state.documents)} document(s) indexed**")
        st.write("")
        
        for doc in st.session_state.documents:
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"📄 **{doc}**")
            
            with col2:
                if st.button("🗑️", key=f"delete_{doc}", help=f"Delete {doc}"):
                    if delete_document(doc):
                        st.success(f"Deleted {doc}")
                        st.session_state.documents = fetch_documents()
                        st.rerun()
    else:
        st.info("No documents indexed yet. Upload a document to get started!")

# RIGHT PANEL: Chat Interface
with right_col:
    st.subheader("💬 Chat Interface")
    
    # Display chat messages
    chat_container = st.container(height=500)
    
    with chat_container:
        if len(st.session_state.messages) == 0:
            st.info("👋 Ask me anything! I can answer questions with or without documents.")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show sources if available (for assistant messages)
                if message["role"] == "assistant" and message.get("sources"):
                    st.caption("📚 Sources used:")
                    sources_text = " • ".join([f"**{s}**" for s in message["sources"]])
                    st.markdown(sources_text)
                
                # Show retrieved chunks in an expander (for assistant messages)
                if message["role"] == "assistant" and message.get("chunks"):
                    with st.expander("🔍 View Retrieved Context"):
                        for i, chunk in enumerate(message["chunks"], 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(chunk)
                            if i < len(message["chunks"]):
                                st.divider()
    
    # Chat input
    query = st.chat_input("What would you like to know?")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=False):
        st.session_state.messages = []
        st.rerun()
    
    if query:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })
        
        # Query the backend
        with st.spinner("Thinking..."):
            try:
                query_payload = {
                    "query": query,
                    "top_k": 3,
                    "generate": True
                }
                query_response = requests.post(
                    f"{BACKEND_URL}/query",
                    json=query_payload
                )
                query_response.raise_for_status()
                result = query_response.json()
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.get("answer", "Sorry, I couldn't generate an answer."),
                    "sources": result.get("sources_used", []),
                    "chunks": result.get("retrieved_chunks", [])
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Query failed: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {str(e)}",
                    "sources": [],
                    "chunks": []
                })
                st.rerun()

