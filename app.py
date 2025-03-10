import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Initialize LangSmith client and tracer
os.environ["LANGCHAIN_API_KEY"]="lsv2_pt_fff51c1b5eba43099"
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="PDFRAG"

# Set page configuration
st.set_page_config(page_title="PDF Chat", layout="wide")
st.title("Chat with your PDF")

# Initialize OpenAI API key input
api_key = st.text_input("Enter your OpenAI API key:", type="password")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])

# Initialize chat interface
query = st.text_input("Ask a question about your PDF:")

def process_pdf(uploaded_file):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Load and process the PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(pages)

    # Create embeddings with tracking
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    vectorstore = FAISS.from_documents(
        splits, 
        embeddings
    )
    
    # Clean up temporary file
    os.unlink(tmp_path)
    
    return vectorstore

def get_response(vectorstore, query, api_key):
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    
    # Create prompt template
    template = """Answer the question based on the following context:
    
    Context: {context}
    
    Question: {question}
    
    Answer: """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Initialize language model with tracking
    llm = ChatOpenAI(
        temperature=0,
        api_key=api_key,
        model="gpt-4o-mini",  # Changed from gpt-4o-mini as it was incorrec
    )
    
    # Create RAG chain with tracking
    rag_chain = (
        {"context": retriever, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Get response
    return rag_chain.invoke(query)

# Main app logic
if uploaded_file and api_key and query:
    try:
        with st.spinner("Processing PDF..."):
            vectorstore = process_pdf(uploaded_file)
        
        with st.spinner("Getting answer..."):
            response = get_response(vectorstore, query, api_key)
            st.write("Answer:", response)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
elif not api_key and (uploaded_file or query):
    st.warning("Please enter your OpenAI API key.")
elif not uploaded_file and query:
    st.warning("Please upload a PDF file first.")