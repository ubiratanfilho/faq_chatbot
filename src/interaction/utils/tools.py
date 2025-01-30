import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool

# import dotenv

# dotenv.load_dotenv()

FAISS_PATH = "data/faiss_index"

### Loading Vector-Store and Retriever Tool
if os.path.exists(FAISS_PATH):
    # Load the existing FAISS index
    vector_store = FAISS.load_local(FAISS_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
else:
    raise FileNotFoundError(f"FAISS index not found at {FAISS_PATH}")

retriever = vector_store.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and returns information about Hotmart.",
)