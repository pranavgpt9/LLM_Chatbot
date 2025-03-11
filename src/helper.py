import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Check if OpenAI API Key is set (if needed elsewhere)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.warning("Warning: OPENAI_API_KEY not set in environment variables.")


# Load PDFs from Directory

def load_pdf_files(data_directory):
    if not os.path.exists(data_directory):
        logging.error(f"Error: Directory '{data_directory}' does not exist.")
        return []

    logging.info(f"Loading PDFs from directory: {data_directory}")

    loader = DirectoryLoader(
        data_directory,
        glob="*.pdf",  # Ensure only PDF files are loaded
        loader_cls=PyMuPDFLoader
    )

    documents = loader.load()

    if not documents:
        logging.error("No PDFs found or failed to load.")
        return []

    logging.info(f"Total {len(documents)} documents loaded successfully.")
    return documents


# Split Text into Chunks

def split_text(documents):
    if not documents:
        logging.error("Error: No documents found to split.")
        return []

    logging.info("Splitting documents into smaller chunks for embedding...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documents)

    logging.info(f"Total {len(text_chunks)} text chunks created.")
    return text_chunks


# Create ChromaDB Vector Store

def create_vector_store(text_chunks, persist_directory="chroma_db/"):
    if not text_chunks:
        logging.error("No text chunks available for vector storage.")
        return None

    logging.info(f"Creating vector store at {persist_directory} with {len(text_chunks)} text chunks.")

    # Load Hugging Face Sentence Transformer model for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        vectorstore = Chroma.from_documents(
            text_chunks,
            embeddings,
            collection_name="pdf_docs",
            persist_directory=persist_directory
        )
        vectorstore.persist()
        logging.info("ChromaDB Vector Store created successfully!")
        return vectorstore

    except Exception as e:
        logging.error(f"Error creating ChromaDB: {str(e)}")
        return None


# Create a Retriever for Searching

def get_retriever(vectorstore, k=3):
    if not vectorstore:
        logging.error("Error: Vector store is not initialized.")
        return None

    logging.info(f"Creating retriever with top {k} search results.")
    return vectorstore.as_retriever(search_kwargs={"k": k})
