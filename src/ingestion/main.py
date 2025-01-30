import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import json
import os
import dotenv

dotenv.load_dotenv()
FAISS_PATH = 'data/faiss_index'

# Check if the FAISS index exists
if os.path.exists(FAISS_PATH):
    # Load the existing FAISS index
    vectorstore = FAISS.load_local(FAISS_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
else:
    loader = WebBaseLoader(
        'https://hotmart.com/pt-br/blog/como-funciona-hotmart',
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("content__body")
                )
            ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Create a new FAISS index
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    # Save the FAISS index for future use
    vectorstore.save_local(FAISS_PATH)