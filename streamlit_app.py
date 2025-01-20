import os 
import shutil
import streamlit as st
from dotenv import load_dotenv
from src.preprocessing import load_documents, split_documents, upload_to_qdrant
from src.retrieve import retrieve_answer_from_docs, clear_qdrant_data
from custom_logger import logger
from exception import CustomException

# Load environment variables
load_dotenv()

# Streamlit UI components
st.title("Document Processing and Querying App")

# Sidebar buttons
st.sidebar.title("Options")
delete_data = st.sidebar.button("Delete Existing Data")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if delete_data:
    try:
        st.write("Deleting existing documents from the database...")

        # Call the function to clear Qdrant data
        qdrant_url = os.getenv('qdrant_url')
        qdrant_api_key = os.getenv('qdrant_api')
        collection_name = "rag"  

        if not all([qdrant_url, qdrant_api_key]):
            raise ValueError("One or more environment variables are missing.")

        cleared = clear_qdrant_data(qdrant_url, qdrant_api_key, collection_name)

        if cleared:
            st.success("All documents have been deleted from the database. Ready for new uploads.")
            st.session_state["question"] = ""
        else:
            st.warning("Collection is already empty or does not exist.")

    except Exception as e:
        st.error(f"Error clearing database: {str(e)}")
        logger.error("Error clearing Qdrant data: %s", str(e))

# File upload section for PDF
if uploaded_file is not None:
    try:
        # Save the uploaded file temporarily
        temp_filename = f"temp_{uploaded_file.name}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(uploaded_file, buffer)

        # Step 1: Load documents from the uploaded PDF
        st.write("Processing the document...")
        documents = load_documents(temp_filename)

        # Step 2: Split the documents into chunks
        chunk_size = 2000  # Adjust as necessary
        chunk_overlap = 400  # Adjust as necessary
        chunks = split_documents(documents, chunk_size, chunk_overlap)

        # Step 3: Upload chunks to Qdrant
        qdrant_url = os.getenv('qdrant_url')
        qdrant_api_key = os.getenv('qdrant_api')

        if not all([qdrant_url, qdrant_api_key]):
            raise ValueError("One or more environment variables are missing.")

        upload_to_qdrant(chunks, qdrant_url, qdrant_api_key)

        # Optionally, remove the temporary file after processing
        os.remove(temp_filename)

        st.success("Document processed and uploaded to Qdrant successfully.")

    except ValueError as ve:
        st.error(f"Error: {str(ve)}")
        logger.error("Environment variable error: %s", ve)
    except CustomException as ce:
        st.error(f"Error: {str(ce)}")
        logger.error("Custom exception occurred: %s", str(ce))
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error("An error occurred: %s", str(e))

# Query section for asking questions
st.subheader("Ask a Question")

question = st.text_input("Enter your question:")

if question:
    try:
        answer = retrieve_answer_from_docs(question)
        st.write(f"Answer: {answer}")

    except CustomException as ce:
        st.warning(str(ce))  # Display a friendly message for CustomException
        logger.warning("Custom exception occurred: %s", str(ce))

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.error("Error retrieving answer: %s", str(e))
