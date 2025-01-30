# FAQ Chatbot

Esse repositório contém um chatbot que responde perguntas sobre a Hotmart. Para isso, foi desenvolvido um sistema de multi-agentes que verifica se a pergunta é geral, ou é específica sobre a Hotmart. Caso seja este o caso, acessa o index criado com o conteúdo da [FAQ](https://hotmart.com/pt-br/blog/como-funciona-hotmart) da Hotmart e gera a resposta.

## Requisitos
 
- Docker
- Docker Compose
- OpenAI API Key

## Como rodar

1. Exporte a sua chave de API da OpenAI

    ```bash
    export OPENAI_API_KEY="your-openai-api-key"
    ```

2. Rode o comando abaixo para subir os serviços

    ```bash
    docker-compose up --build
    ```

3. Faça a requisição para criar o índice:

    ```bash
    curl --location 'http://localhost:5000/criar_index' \
    --header 'Content-Type: application/json' \
    --data '{
        "url": "https://hotmart.com/pt-br/blog/como-funciona-hotmart",
        "chunk_size": 1000,
        "chunk_overlap": 200
    }'
    ```

4. Faça a requisição para gerar a resposta:

    ```bash
    curl --location 'http://localhost:5001/generate_answer' \
    --header 'Content-Type: application/json' \
    --data '{
        "question": "Oi, o que a Hotmart faz",
        "thread_id": 1
    }'
    ```