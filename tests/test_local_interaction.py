from utils.graph import graph
from dotenv import load_dotenv

load_dotenv()

question = "Como ganhar dinheiro com a Hotmart?"
config = {"configurable": {"thread_id": "123"}}

response = graph.invoke({"messages": [("user", question)], "question": question}, config)

print(response['messages'][-1].content)