from flask import Flask, render_template, request, jsonify
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings
from src.helper import get_retriever
from src.prompt import get_llm_prompt

app = Flask(__name__)
CORS(app)  # Enable CORS to prevent frontend issues

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Load embeddings
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the vector store with embeddings
vectorstore = Chroma(persist_directory="chroma_db/", embedding_function=embedding_function)

# Create the retriever
retriever = get_retriever(vectorstore)

# Instantiate LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.4,
    max_tokens=500,
    api_key=OPENAI_API_KEY
)

# Create RAG chain
llm_prompt = get_llm_prompt()
chain = create_stuff_documents_chain(llm, llm_prompt)
RAG = create_retrieval_chain(retriever, chain)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chatbot():
    msg = request.form.get("msg")  # Get user message from POST request
    if not msg:
        return jsonify({"error": "No input received"}), 400

    logging.info(f"User Input: {msg}")
    
    try:
        response = RAG.invoke({"input": msg})
        chatbot_response = response.get("answer", "I'm sorry, I couldn't process that.")
    except Exception as e:
        logging.error(f"Error processing response: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

    logging.info(f"Chatbot Response: {chatbot_response}")

    return jsonify({"response": chatbot_response})  # Ensure response is in JSON format

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

#handles out-of-context questions by retrieving relevant information

# @app.route("/get", methods=["POST"])
# def chatbot():
#     msg = request.form.get("msg")  # Get user message from POST request
#     if not msg:
#         return jsonify({"error": "No input received"}), 400

#     logging.info(f"User Input: {msg}")

#     try:
#         # Retrieve context-relevant documents
#         docs = retriever.invoke(msg)

#         # Log the number of retrieved documents
#         logging.info(f"Retrieved {len(docs)} documents for input: {msg}")


#         # Optionally log details of each document (ensure no sensitive info is logged)
#         for idx, doc in enumerate(docs):
#             logging.debug(f"Document {idx}: {doc}")

#         # Check if any documents were retrieved
#         if not docs:
#             return jsonify({"response": "I'm sorry, I can only answer context-based questions."})

#         # Generate response using RAG
#         response = RAG.invoke({"input": msg})
#         chatbot_response = response.get("answer", "I'm sorry, I couldn't process that.")

#         # Compare the response with the retrieved documents
#         retrieved_text = " ".join([doc.page_content for doc in docs])
#         if chatbot_response.strip() in retrieved_text:
#             logging.info("Response is based on retrieved context.")
#         else:
#             logging.info("Response is out of context.")
#             chatbot_response = "Your question is out of context. Please ask a context-based question."

#     except Exception as e:
#         logging.error(f"Error processing response: {str(e)}")
#         return jsonify({"error": "Internal server error"}), 500

#     logging.info(f"Chatbot Response: {chatbot_response}")
#     return jsonify({"response": chatbot_response})


# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=8080, debug=True)