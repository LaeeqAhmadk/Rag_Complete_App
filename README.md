# Rag_Complete_App

Rag_Complete_App is a streamlined and user-friendly application designed to implement Retrieval-Augmented Generation (RAG). It supports custom document ingestion, preprocessing, and querying with Qdrant, making it ideal for creating intelligent and context-aware solutions.

## Features

1. Document Uploading and Preprocessing

- Easily upload and preprocess documents.

- Supports splitting documents into manageable chunks for effective retrieval.

2. Integration with Qdrant

- Indexes and retrieves documents using Qdrant, a powerful vector database.

- Efficient storage and retrieval of vectorized document embeddings.

3. Streamlit Interface

- Intuitive interface built with Streamlit.

- Seamlessly upload, query, and retrieve relevant information.

4. Extensibility

- Modular design for easy integration and customization.

- Supports enhancements to improve retrieval and generation workflows.

## Project Structure

Rag_Complete_App/
├── src/
│   ├── streamlit_app.py       # Main application file
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── load_documents.py  # Handles document loading
│   │   ├── split_documents.py # Splits documents into chunks
│   │   ├── upload_to_qdrant.py # Uploads embeddings to Qdrant
│   ├── retrieve.py            # Handles document retrieval
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation (this file)

Setup Instructions

Prerequisites

Python 3.9 or above.

Qdrant installed and running.

Streamlit installed.

Installation

Clone the repository:

git clone https://github.com/yourusername/Rag_Complete_App.git
cd Rag_Complete_App

Install dependencies:

pip install -r requirements.txt

Start Qdrant (if running locally):

docker run -p 6333:6333 qdrant/qdrant

Launch the app:

streamlit run src/streamlit_app.py

Usage

Open the app in your browser at http://localhost:8501.

Upload documents through the interface.

Preprocess and split documents.

Query the system to retrieve relevant information using the Qdrant backend.

Current Focus

Enhancing the interface of the application for better user experience.

Fixing issues in retrieve.py, including defining the format_docs function.

Preparing the application for deployment on Hugging Face.

Future Enhancements

Optimization: Fine-tune retrieval and embedding processes.

Deployment: Host the application on Hugging Face Spaces.

Advanced Features: Add support for more vector databases and improve RAG workflows.

Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

License

This project is licensed under the Apache 2.0 License. See LICENSE for more details.

Acknowledgments

Special thanks to the creators of Qdrant, Streamlit, and the open-source community for their tools and support.

