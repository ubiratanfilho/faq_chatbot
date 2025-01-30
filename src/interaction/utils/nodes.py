from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langgraph.graph import MessagesState

from utils.tools import retriever_tool

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
    model = model.bind_tools([retriever_tool])
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