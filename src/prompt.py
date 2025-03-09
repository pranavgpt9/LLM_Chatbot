from langchain_core.prompts import ChatPromptTemplate

def get_llm_prompt():
    prompt = (
        "You are a knowledgeable question-answering assistant. "
        "Use the following retrieved context to answer the question accurately. "
        "If you don't have enough information, simply state that you don't know. "
        "Keep your response concise—no more than three sentences.\n\n"
        "{context}"
    )
    return ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("human", "{input}")
    ])