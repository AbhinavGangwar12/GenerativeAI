# mcp_server.py
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("Local-Developer-Tools")

# Instantiating the Search Run tool locally
search_tool = DuckDuckGoSearchRun(region="us-en")


# --- REGISTER TOOLS VIA MCP ---


@mcp.tool()
def calculator(
    first_number: float, second_number: float, operation: str
) -> dict:
    """Perform a basic arithmetic operation on two numbers.

    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_number + second_number
        elif operation == "sub":
            result = first_number - second_number
        elif operation == "mul":
            result = first_number * second_number
        elif operation == "div":
            result = first_number / second_number
        else:
            return {"error": "Unsupported operation"}

        return {
            "first_number": first_number,
            "second_number": second_number,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_stock_price(symbol: str) -> dict:
    """Fetch the current stock price for the given stock symbol using the Alpha Vantage API."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=N67JWRVPZ21AZO3N"
    r = requests.get(url)
    return r.json()


@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for up-to-date information regarding a given query string."""
    return search_tool.run(query)


if __name__ == "__main__":
    # Start the server using standard I/O transport
    mcp.run(transport="stdio")