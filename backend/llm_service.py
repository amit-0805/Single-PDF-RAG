from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from logger import setup_logger

logger = setup_logger()

def get_llm(model_type: str, model_name: str, api_key: str, temperature: float, max_tokens: int):
    try:
        logger.info(f"Initializing {model_type} model: {model_name}")
        
        if model_type == "openai":
            return ChatOpenAI(
                temperature=temperature,
                api_key=api_key,
                model=model_name,
                max_tokens=max_tokens
            )
        elif model_type == "groq":
            return ChatGroq(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        raise

def get_response(llm, vectorstore, query: str) -> str:
    try:
        logger.info("Generating response")
        
        template = """Answer the question properly and if the context is not present just tell not present in the PDF Uploaded and then if it's present then based on the following context:
        
        Context: {context}
        Question: {question}
        Answer: """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        rag_chain = (
            {"context": retriever, "question": lambda x: x}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = rag_chain.invoke(query)
        logger.info("Response generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise