import os
from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages, StateGraph
from typing_extensions import TypedDict

load_dotenv()

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

graph_builder = StateGraph(State)

llm = init_chat_model(
    "openai:gpt-5-mini",
    temperature=0.6,
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()