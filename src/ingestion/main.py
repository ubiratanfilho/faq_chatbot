from flask import Flask, request, jsonify
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

app = Flask(__name__)

FAISS_PATH = 'data/faiss_index'

@app.route('/criar_index', methods=['POST'])
def criar_index():
    data = request.get_json()
    url = data.get('url')
    openai_api_key = data.get('openai_api_key')
    chunk_size = data.get('chunk_size', 1000)
    chunk_overlap = data.get('chunk_overlap', 200)

    if not url:
        return jsonify({"error": "Parâmetro 'url' não fornecido."}), 400
    if not openai_api_key:
        return jsonify({"error": "Parâmetro 'openai_api_key' não fornecido."}), 400

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
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(FAISS_PATH)

    return jsonify({"message": "FAISS index carregado ou criado com sucesso!", "url": url})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
