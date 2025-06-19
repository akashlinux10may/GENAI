import os
import asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from langchain_ollama import OllamaLLM

# Load environment variables from .env
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Missing GITHUB_TOKEN in environment.")

# Define MCP server parameters using Docker
server_params = StdioServerParameters(
    command="docker",
    args=[
        "run", "-i", "--rm",
        "-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={GITHUB_TOKEN}",
        "ghcr.io/github/github-mcp-server"
    ]
)

# Use LLaMA 3 from Ollama
llm = OllamaLLM(model="llama3")

# Callback for responding with LLaMA model
async def handle_sampling_message(message: types.CreateMessageRequestParams) -> types.CreateMessageResult:
    input_text = getattr(message.content, "text", "")
    response = await asyncio.to_thread(llm.invoke, input_text)
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text=response),
        model="llama3",
        stopReason="endTurn"
    )

# Main function
async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, sampling_callback=handle_sampling_message) as session:
            print("\n✅ Connected to MCP Server")

            await session.initialize()

            print("\n🔍 Searching repositories for: 'akashlinux10may'")
            result = await session.call_tool(
                "search_repositories",
                {"query": "user:akashlinux10may", "per_page": 5, "page": 1}
            )

            if result.isError:
                print("Search failed:")
                for msg in result.content:
                    print("  ↳", msg.text)
            else:
                print("Repositories found:")
                for msg in result.content:
                    print("  🔍", msg.text)


if __name__ == "__main__":
    asyncio.run(run())