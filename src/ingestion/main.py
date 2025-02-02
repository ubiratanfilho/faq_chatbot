from flask import Flask, request, jsonify

from utils.ingestion import ingest
from utils.consts import FAISS_PATH

app = Flask(__name__)

@app.route('/criar_index', methods=['POST'])
def criar_index():
    data = request.get_json()
    url = data.get('url')
    chunk_size = data.get('chunk_size', 1000)
    chunk_overlap = data.get('chunk_overlap', 200)

    if not url:
        return jsonify({"error": "Parâmetro 'url' não fornecido."}), 400

    # Ingest data
    ingest(url, FAISS_PATH, chunk_size, chunk_overlap)

    return jsonify({"message": "FAISS index carregado ou criado com sucesso!", "url": url})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
