import os

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .consts import FAISS_PATH

def retriever(state):
    """
    Node to retrieve relevant documents.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---RETRIEVER---")
    
    # Get state
    question = state["question"]
    
    ### Loading Vector-Store and Retriever Tool
    if os.path.exists(FAISS_PATH):
        # Load the existing FAISS index
        vector_store = FAISS.load_local(FAISS_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:
        raise FileNotFoundError(f"FAISS index not found at {FAISS_PATH}")

    retriever = vector_store.as_retriever()

    # Retrieve documents
    context = retriever.invoke(question)
    
    return {"context": context}

def assistant(state):
    """
    Node to generate a response to a question based on the given context.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """
    print("---ASSISTANT---")
    
    # Get state
    messages = state["messages"]
    context = state["context"]

    # Chain
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a support assistant from the Hotmart platform for question-answering tasks."
                "Use the following pieces of retrieved context to answer the question."
                "If you don't know the answer, just say that you don't know."
                "Use three sentences maximum and keep the answer concise."
                "Context: {context}"
                "Message History (answer the latest question):"
            ),
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    chain = prompt | llm
    
    # Generate response
    answer = chain.invoke({"messages": messages, "context": context})
    
    return {"messages": [answer]}