# frontend/app.py
import streamlit as st
import requests
import json
from datetime import datetime

# Constants
BACKEND_URL = "http://localhost:8000"
OPENAI_MODELS = ["gpt-4", "gpt-4o", "gpt-4o-mini"]
GROQ_MODELS = ["Gemma2-9b-It", "Deepseek-R1-Distill-Llama-70b", "Mixtral-8x7b-32768"]

def initialize_session_state():
    """Initialize session state variables"""
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}

def create_new_chat(file_id):
    """Create a new chat session for a specific PDF"""
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = st.session_state.uploaded_files[file_id]["name"]
    st.session_state.chats[chat_id] = {
        "messages": [],
        "title": f"Chat about {file_name}",
        "timestamp": datetime.now(),
        "file_id": file_id
    }
    st.session_state.current_chat_id = chat_id

def start_new_chat():
    """Reset current chat to start a new one"""
    st.session_state.current_chat_id = None

def display_chat_history():
    """Display chat messages from current chat"""
    if st.session_state.current_chat_id:
        for message in st.session_state.chats[st.session_state.current_chat_id]["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

def main():
    st.set_page_config(page_title="PDF Chat", layout="wide")
    
    initialize_session_state()
    
    # Sidebar with three sections
    with st.sidebar:
        st.title("PDF Chat Settings")
        
        # New Chat Button at the top of sidebar
        if st.button("➕ New Chat", type="primary", use_container_width=True):
            start_new_chat()
            st.rerun()
        
        st.divider()
        
        # Section 1: Model Configuration
        with st.expander("🤖 Model Configuration", expanded=True):
            model_type = st.radio("Select Model Provider:", ["OpenAI", "Groq"])
            
            if model_type == "OpenAI":
                api_key = st.text_input("OpenAI API Key:", type="password")
                model_name = st.selectbox("Select Model:", OPENAI_MODELS)
            else:
                api_key = st.text_input("Groq API Key:", type="password")
                model_name = st.selectbox("Select Model:", GROQ_MODELS)
            
            temperature = st.slider(
                "Temperature:",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )
            st.text("Temperature: Temperature tells how much creative the model gets in generating output")
            
            max_tokens = st.slider(
                "Max Tokens:",
                min_value=100,
                max_value=4096,
                value=2048,
                step=100
            )
            st.text("max_tokens: Limits the maximum number of tokens the model can generate in output")
        
        # Section 2: Uploaded Files Reference
        with st.expander("📚 Uploaded PDFs", expanded=True):
            if not st.session_state.uploaded_files:
                st.info("No PDFs uploaded yet")
            else:
                st.write("Your uploaded PDFs:")
                for file_id, file_data in st.session_state.uploaded_files.items():
                    st.write(f"📄 {file_data['name']}")
        
        # Section 3: Chat History
        with st.expander("💬 Chat History", expanded=True):
            if st.session_state.chats:
                for chat_id, chat_data in sorted(
                    st.session_state.chats.items(),
                    key=lambda x: x[1]["timestamp"],
                    reverse=True
                ):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if st.button(
                            chat_data["title"],
                            key=f"chat_{chat_id}",
                            use_container_width=True
                        ):
                            st.session_state.current_chat_id = chat_id
                            st.rerun()
                    with col2:
                        if st.button("🗑️", key=f"delete_{chat_id}"):
                            del st.session_state.chats[chat_id]
                            if chat_id == st.session_state.current_chat_id:
                                st.session_state.current_chat_id = None
                            st.rerun()
    
    # Main area
    if not st.session_state.current_chat_id:
        st.title("Chat with your PDF")
        
        # File upload or selection section
        uploaded_file = st.file_uploader("Upload a new PDF", type=['pdf'])
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                try:
                    files = {"file": uploaded_file}
                    response = requests.post(f"{BACKEND_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        file_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.session_state.uploaded_files[file_id] = {
                            "name": uploaded_file.name,
                            "timestamp": datetime.now()
                        }
                        create_new_chat(file_id)
                        st.success("PDF processed successfully!")
                        st.rerun()
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {str(e)}")
        
        # Start new chat with existing PDF
        if st.session_state.uploaded_files:
            st.write("Or start a new chat with an existing PDF:")
            for file_id, file_data in st.session_state.uploaded_files.items():
                if st.button(f"Chat about {file_data['name']}", key=f"start_chat_{file_id}"):
                    create_new_chat(file_id)
                    st.rerun()
    
    else:
        # Active chat interface
        current_chat = st.session_state.chats[st.session_state.current_chat_id]
        file_name = st.session_state.uploaded_files[current_chat["file_id"]]["name"]
        
        # Show current PDF context
        st.title(current_chat["title"])
        st.info(f"📄 Currently chatting about: {file_name}")
        
        # Chat interface
        display_chat_history()
        
        # Chat input
        if query := st.chat_input("Ask a question about your PDF:"):
            if not api_key:
                st.warning("Please enter an API key.")
                return
            
            # Add user message to chat history
            current_chat["messages"].append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            
            # Get response from backend
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        chat_request = {
                            "query": query,
                            "api_key": api_key,
                            "model_type": model_type.lower(),
                            "model_name": model_name,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "session_id": st.session_state.current_chat_id
                        }
                        
                        response = requests.post(
                            f"{BACKEND_URL}/chat",
                            json=chat_request,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            assistant_response = response.json()["response"]
                            st.markdown(assistant_response)
                            current_chat["messages"].append({
                                "role": "assistant",
                                "content": assistant_response
                            })
                        else:
                            error_detail = response.json().get('detail', 'Unknown error')
                            st.error(f"Error: {error_detail}")
                    except requests.exceptions.Timeout:
                        st.error("Request timed out. Please try again.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error connecting to backend: {str(e)}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()