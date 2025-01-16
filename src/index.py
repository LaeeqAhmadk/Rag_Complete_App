import os
import sys
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer  # Open-source embedding model
from langchain_community.vectorstores import Qdrant
from qdrant_client.http.models import BinaryQuantization, BinaryQuantizationConfig
from qdrant_client import QdrantClient
from custom_logger import logger
from exception import CustomException

# Load environment variables from a .env file
load_dotenv()

def store_documents_to_qdrant(texts: list):
    """
    Store documents into Qdrant vector store.

    Args:
        texts (list): List of text chunks.

    Returns:
        Qdrant: Qdrant vector store instance.
    """
    try:
        # Fetch configuration details from environment variables
        qdrant_end = os.getenv('qdrant_end')
        qdrant_api_key = os.getenv('qdrant_api')

        if not qdrant_end or not qdrant_api_key:
            raise ValueError("Environment variables QDRANT_END and QDRANT_API_KEY must be set.")

        # Initialize the Sentence-Transformers model (open-source)
        embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose different models from sentence-transformers

        # Configure Qdrant with quantization settings
        qdrant = Qdrant.from_documents(
            texts=texts,
            embedding=embeddings_model.encode,  # Pass the encode function to get embeddings
            url=qdrant_end,
            prefer_grpc=True,
            api_key=qdrant_api_key,
            collection_name="Lex-v1",
            quantization_config=BinaryQuantization(
                binary=BinaryQuantizationConfig(always_ram=True)
            )
        )

        logger.info("Documents stored in Qdrant successfully.")
        return qdrant
    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
        raise CustomException(ve, sys)
    except Exception as e:
        logger.error(f"An error occurred while storing documents in Qdrant: {e}")
        raise CustomException(e, sys)
