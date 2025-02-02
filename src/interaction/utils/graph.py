from typing import List

from langgraph.graph import START, StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import BaseMessage

from utils.nodes import retriever, assistant
from .consts import RETRIEVER, ASSISTANT


# States
class State(MessagesState):
    question: str
    context: List[str]

# Graph
builder = StateGraph(State)

builder.add_node(RETRIEVER, retriever)
builder.add_node(ASSISTANT, assistant)

builder.add_edge(START, RETRIEVER)
builder.add_edge(RETRIEVER, ASSISTANT)
builder.add_edge(ASSISTANT, END)

graph = builder.compile(checkpointer=MemorySaver())

with open("images/graph.png", "wb") as f:
    f.write(graph.get_graph(xray=True).draw_mermaid_png())