import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

load_dotenv()

llm = init_chat_model(
    "openai:qwen-max",
    temperature=0,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
tavily_tool = TavilySearch(max_results=5)

async def make_graph():
    client = MultiServerMCPClient(
        {
            "map": {
                "url": f"https://mcp.amap.com/mcp?key={os.getenv('GAODE_MAP_KEY')}",
                "transport": "streamable_http",
            }
        }
    )
    tools = [tavily_tool, human_assistance] + await client.get_tools()
    return create_react_agent(llm, tools)