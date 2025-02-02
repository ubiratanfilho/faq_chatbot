import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
    
def ingest(url: str, FAISS_PATH: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
    """
    Cria um vetor FAISS a partir do conteúdo de um site e salva em disco.
    """
    # Carrega o conteúdo do site a partir da URL recebida
    loader = WebBaseLoader(
        url,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_="content__body")
        ),
    )
    docs = loader.load()

    # Divide o texto em chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)

    # Cria e salva o vetor FAISS usando a chave fornecida
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(FAISS_PATH)