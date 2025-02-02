from flask import Flask, request, jsonify
from utils.graph import graph

app = Flask(__name__)

FAISS_PATH = 'data/faiss_index'

@app.route('/generate_answer', methods=['POST'])
def criar_index():
    data = request.get_json()
    question = data.get('question')
    thread_id = data.get('thread_id')
    
    if not question:
        return jsonify({"error": "Parâmetro 'question' não fornecido."}), 400

    config = {"configurable": {"thread_id": thread_id}}
    response = graph.invoke({"messages": [("user", question)], "question": question}, config)

    return jsonify({"message": response['messages'][-1].content})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)