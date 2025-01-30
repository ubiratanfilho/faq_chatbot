from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document

from typing_extensions import List, TypedDict
import os
import dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

dotenv.load_dotenv()

FAISS_PATH = "data/faiss_index"

# Check if the FAISS index exists
if os.path.exists(FAISS_PATH):
    # Load the existing FAISS index
    vector_store = FAISS.load_local(FAISS_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
else:
    raise FileNotFoundError(f"FAISS index not found at {FAISS_PATH}")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define prompt for question-answering
prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Você é um assistente que tem acesso a documentos da empresa Hotmart. 
        Responda apenas questões relacionadas à Hotmart. 
        Utilize o contexto abaixo para responder à pergunta:
        {context}
    """),
    ("human", "{question}")
]
)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

if __name__ == "__main__":
    response = graph.invoke({"question": "O que é a Hotmart?"})
    print(response["answer"])