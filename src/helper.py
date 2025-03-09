import os
import logging
from dotenv import load_dotenv
# from langchain.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

def load_pdf_files(data_directory):
    loader = DirectoryLoader(
        data_directory,
        glob="*.pdf",  # Ensure only PDF files are loaded
        loader_cls=PyMuPDFLoader
    )
    return loader.load()

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(documents)

def create_vector_store(text_chunks, persist_directory="chroma_db/"):
    # Use Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        text_chunks,
        embeddings,
        collection_name="pdf_docs",
        persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore

def get_retriever(vectorstore, k=3):
    return vectorstore.as_retriever(search_kwargs={"k": k})
