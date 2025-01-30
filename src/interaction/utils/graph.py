from langgraph.graph import START, StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from utils.nodes import agent, generate
from utils.tools import retriever_tool

import dotenv

dotenv.load_dotenv()

### Defining the graph
workflow = StateGraph(MessagesState)

workflow.add_node("agent", agent)
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile(checkpointer=MemorySaver()) 

# Saving the graph as an image
with open("images/graph.png", "wb") as f:
    f.write(graph.get_graph(xray=True).draw_mermaid_png())