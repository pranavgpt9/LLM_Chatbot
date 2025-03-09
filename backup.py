import os
import logging
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from src.helper import get_retriever
# from src.helper import create_vector_store
# from src.helper import text_chunks
from src.prompt import get_llm_prompt



app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# embedding = create_vector_store(text_chunks)


# Load the vector store and create a retriever
vectorstore = Chroma(persist_directory="chroma_db/")
retriever = get_retriever(vectorstore)

# Instantiate the LLM using LangChain's ChatOpenAI
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.4,
    max_tokens=500,
    api_key=OPENAI_API_KEY  # Pass the API key directly
)

# Define the prompt template
llm_prompt = get_llm_prompt()

# Create the RAG chain
chain = create_stuff_documents_chain(llm, llm_prompt)
RAG = create_retrieval_chain(retriever, chain)

# Example query
# response = RAG.invoke({"input": "What vaccinations should be given to babies up to 6 months old?"})
# print(response["answer"])

@app.route("/")
def redirect():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chatbot():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = RAG.invoke({"input":msg})
    print("Response :", response["answer"])
    return str(response["answer"])


if __name__ =='__main__':
    app.run(host="0.0.0.0", port = 8080, debug=True)