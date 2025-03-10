from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from pdf_processor import process_pdf
from llm_service import get_llm, get_response
from logger import setup_logger

logger = setup_logger()
app = FastAPI()

vectorstore = None

class ChatRequest(BaseModel):
    query: str
    api_key: str
    model_type: str
    model_name: str
    temperature: float
    max_tokens: int
    session_id: str

@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        global vectorstore
        content = await file.read()
        vectorstore = process_pdf(content)
        
        return {"message": "PDF processed successfully"}
        
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if not vectorstore:
            raise HTTPException(status_code=400, detail="Please upload a PDF first")
        
        llm = get_llm(
            request.model_type,
            request.model_name,
            request.api_key,
            request.temperature,
            request.max_tokens
        )
        
        response = get_response(llm, vectorstore, request.query)
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)