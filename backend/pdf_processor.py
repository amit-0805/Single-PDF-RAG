from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os
from logger import setup_logger

logger = setup_logger()

def process_pdf(file_content: bytes) -> FAISS:
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
            
        logger.info(f"Processing PDF file")
        
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
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create vector store
        vectorstore = FAISS.from_documents(splits, embeddings)
        logger.info("PDF processed successfully")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise
        
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info("Temporary PDF file cleaned up")