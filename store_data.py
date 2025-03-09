import logging
from src.helper import load_pdf_files, split_text, create_vector_store


def main():
    # Load and process documents
    documents = load_pdf_files("C:\\Users\\prana\\Projects AI\\LLM_Projects\\DATA")
    text_chunks = split_text(documents)
    logging.info(f"Number of text chunks: {len(text_chunks)}")

    # Create and persist the vector store
    create_vector_store(text_chunks)

if __name__ == "__main__":
    main()