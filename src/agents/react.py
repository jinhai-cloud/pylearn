import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = init_chat_model(
    "openai:qwen-max",
    temperature=0,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

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
    tools = await client.get_tools()
    tools.append(tavily_tool)
    return create_react_agent(llm, tools)