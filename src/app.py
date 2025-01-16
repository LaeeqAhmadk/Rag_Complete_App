import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from src.preprocessing import load_documents, split_documents, upload_to_qdrant
from src.retrieve import retrieve_answer_from_docs
from custom_logger import logger
from exception import CustomException
import shutil

load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    """Schema for query request validation."""
    question: str

@app.post("/process/")
async def process_documents(file: UploadFile = File(...)):
    """
    Endpoint to process and upload documents to Qdrant.
    
    Accepts a PDF file upload, processes it, and uploads the document chunks to Qdrant.
    """
    try:
        # Load environment variables
        qdrant_url = os.getenv('qdrant_url')
        qdrant_api_key = os.getenv('qdrant_api')
        
        if not all([qdrant_url, qdrant_api_key]):
            raise ValueError("One or more environment variables are missing.")
        
        # Save the uploaded file temporarily
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 1: Load documents from the uploaded PDF
        documents = load_documents(temp_filename)
        
        # Step 2: Split the documents into chunks
        chunk_size = 2000  # Adjust as necessary
        chunk_overlap = 400  # Adjust as necessary
        chunks = split_documents(documents, chunk_size, chunk_overlap)
        
        # Step 3: Upload chunks to Qdrant
        upload_to_qdrant(chunks, qdrant_url, qdrant_api_key)

        # Optionally, remove the temporary file after processing
        os.remove(temp_filename)

        return {"message": "Document processed and uploaded to Qdrant successfully."}

    except ValueError as ve:
        logger.error("Environment variable error: %s", ve)
        raise HTTPException(status_code=400, detail=str(ve))
    except CustomException as ce:
        logger.error("Custom exception occurred: %s", str(ce))
        raise HTTPException(status_code=500, detail=str(ce))
    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.post("/ask/")
async def ask_question(query: QueryRequest):
    """
    Endpoint to retrieve the answer from documents based on a query.

    Args:
        query (QueryRequest): The input query containing a question string.

    Returns:
        dict: A response containing the input question and the generated answer.
    """
    try:
        answer = retrieve_answer_from_docs(query.question)
        return {"question": query.question, "answer": answer}

    except Exception as e:
        logger.error("Error retrieving answer: %s", str(e))
        raise HTTPException(status_code=500, detail="An error occurred while retrieving the answer.")

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI app using Uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
