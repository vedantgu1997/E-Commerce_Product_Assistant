from mcp.server.fastmcp import FastMCP
from product_assistant.retriever.retrieval import Retriever
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize the MCP server
mcp = FastMCP("hybrid_search")

# Load Retriever
retriever_instance = Retriever()
retriever = retriever_instance.load_retriever()

# Langchain DuckDuckGo Search Tool
web_search_tool = DuckDuckGoSearchRun()

# ---------------- Helpers ----------------
def format_docs(docs) -> str:
    if not docs:
        return ""
    formatted_chunks = []
    for d in docs:
        meta = d.metadata or {}
        formatted = (
            f"Title: {meta.get('product_title', 'N/A')}\n"
            f"Price: {meta.get('price', 'N/A')}\n"
            f"Rating: {meta.get('rating', 'N/A')}\n"
            f"Reviews: \n{d.page_content.strip()}"
        )
        formatted_chunks.append(formatted)
    return "\n\n---\n\n".join(formatted_chunks)

# ---------------- MCP Tools ----------------
@mcp.tool()
async def get_product_info(query: str) -> str:
    """Fetch product info from vector DB."""
    try:
        docs = retriever.invoke(query)  # type: ignore
        context = format_docs(docs)
        if not context:
            return "No local results found."
        return context
    except Exception as e:
        return f"Error retrieving product info: {str(e)}"
    
@mcp.tool()
async def web_search(query: str) -> str:
    """Perform web search using DuckDuckGo."""
    try:
        result = web_search_tool.run(query)
        return result
    except Exception as e:
        return f"Error performing web search: {str(e)}"
    
# Run the MCP server
if __name__ == "__main__":
    mcp.run(transport="streamable-http")