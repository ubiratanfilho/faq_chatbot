from typing import List

from langgraph.graph import START, StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import BaseMessage

from utils.nodes import retriever, assistant, need_for_retrieve
from .consts import RETRIEVER, ASSISTANT, NEED_FOR_RETRIEVAL


# States
class State(MessagesState):
    question: str
    context: List[str]
    need_for_context: bool

# Graph
builder = StateGraph(State)

builder.add_node(NEED_FOR_RETRIEVAL, need_for_retrieve)
builder.add_node(RETRIEVER, retriever)
builder.add_node(ASSISTANT, assistant)

def conditional_retrieve(state: List[BaseMessage]) -> str:
    if state["need_for_context"]:
        return RETRIEVER
    return ASSISTANT

builder.add_edge(START, NEED_FOR_RETRIEVAL)
builder.add_conditional_edges(NEED_FOR_RETRIEVAL, conditional_retrieve)
builder.add_edge(RETRIEVER, ASSISTANT)
builder.add_edge(ASSISTANT, END)

graph = builder.compile(checkpointer=MemorySaver())

with open("images/graph.png", "wb") as f:
    f.write(graph.get_graph(xray=True).draw_mermaid_png())