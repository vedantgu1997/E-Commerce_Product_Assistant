import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    client = MultiServerMCPClient({
        "hybrid_search": {
            "command": "python",
            "args": ["/Users/vedantgupta/Desktop/E-Commerce_Product_Assistant/product_assistant/mcp_servers/product_search_server.py"],
            "transport": "stdio"
        }
    })

    # Discover tools
    tools = await client.get_tools()
    print("Discovered tools: ", [t.name for t in tools])

    # Pick tools by name
    retriever_tool = next(t for t in tools if t.name == "get_product_info")
    web_tool = next(t for t in tools if t.name == "web_search")

    # Step 1: Try retriever first
    query = "iPhone 17?"
    retriever_result = await retriever_tool.invoke(query)
    print("\nRetriever result: \n", retriever_result)

    # Step 2: Fall back to web search if retriever fails
    if not retriever_result.strip() or "No local results found" in retriever_result:
        print("\nFalling back to web search...")
        web_result = await web_tool.ainvoke(query)
        print("\nWeb search result: \n", web_result)

if __name__ == "__main__":
    asyncio.run(main())