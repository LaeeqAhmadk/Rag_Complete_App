import os
import sys
from dotenv import load_dotenv
from src.preprocessing import load_documents, split_documents
from src.index import store_documents_to_qdrant
from src.retrieve import retrieve_answer_from_docs
from src.utils import format_docs, create_embeddings
from custom_logger import logger
from exception import CustomException

# Load environment variables from the .env file
load_dotenv()

def retriever(question: str):
    """
    Retrieve an answer to a question from the indexed documents.

    Args:
        question (str): The input question for retrieval.

    Returns:
        str: The retrieved answer.
    """
    try:
        logger.info("Starting retrieval process...")

        # --- Step 1: Load and preprocess documents ---
        file_path = os.getenv('DOCUMENTS_PATH')  # Set the path to your documents in the .env file
        if not file_path:
            raise ValueError("DOCUMENTS_PATH environment variable is not set.")

        logger.info(f"Loading documents from {file_path}...")
        documents = load_documents(file_path)

        logger.info("Splitting documents into smaller chunks...")
        texts = split_documents(documents)  # Split documents into smaller, manageable chunks

        # --- Step 2: Convert text chunks to embeddings ---
        logger.info("Creating embeddings for document chunks...")
        embeddings = create_embeddings(texts)  # Convert text chunks to word embeddings

        # --- Step 3: Store embeddings in Qdrant ---
        logger.info("Storing embeddings in Qdrant vector database...")
        qdrant = store_documents_to_qdrant(texts, embeddings)

        # --- Step 4: Retrieve the answer ---
        logger.info(f"Retrieving answer for the question: {question}")
        answer = retrieve_answer_from_docs(question)

        logger.info(f"Answer retrieved: {answer}")
        return answer

    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
        raise CustomException(ve, sys)
    except Exception as e:
        logger.error(f"An error occurred during the retrieval process: {e}")
        raise CustomException(e, sys)

# Example usage (For testing purposes)
if __name__ == "__main__":
    try:
        # Example question
        sample_question = "Who is Mike?"
        response = retriever(sample_question)
        print(f"Question: {sample_question}\nAnswer: {response}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
