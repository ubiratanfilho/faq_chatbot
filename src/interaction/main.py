from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

from langgraph.graph import START, StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from typing_extensions import List, TypedDict

from typing import Annotated, Sequence

import os

import dotenv

dotenv.load_dotenv()

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
tools = [retriever_tool]

### Defining Nodes
def agent(state: MessagesState):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---GENERAL AGENT---")
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    model = model.bind_tools(tools)
    response = model.invoke(messages)

    return {"messages": [response]}

def generate(state: MessagesState):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """
    print("---RAG AGENT---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content

    # Chain
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({"context": docs, "question": question})
    
    return {"messages": [response]}

### Defining the graph
workflow = StateGraph(MessagesState)

workflow.add_node("agent", agent)
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)
workflow.add_node(
    "generate", generate
)
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile(checkpointer=MemorySaver()) # 

# Saving the graph as an image
with open("images/graph.png", "wb") as f:
    f.write(graph.get_graph(xray=True).draw_mermaid_png())

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    while True:
        question = input("Usuário: ")
        response = graph.invoke({"messages": [("user", question)]}, config) # config
        print("Bot:", response['messages'][-1].content)