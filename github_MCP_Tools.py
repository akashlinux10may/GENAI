import os
import asyncio
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Missing GITHUB_TOKEN in environment.")

# Set up Docker MCP server
server_params = StdioServerParameters(
    command="docker",
    args=[
        "run", "-i", "--rm",
        "-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={GITHUB_TOKEN}",
        "ghcr.io/github/github-mcp-server"
    ]
)

# Initialize LLaMA 3 via Ollama
llm = OllamaLLM(model="llama3")


# Sampling callback — sends messages to LLaMA 3 and returns the response
async def handle_sampling_message(
    message: types.CreateMessageRequestParams,
) -> types.CreateMessageResult:
    input_text = message.content.text if hasattr(message.content, "text") else ""
    response = await asyncio.to_thread(llm.invoke, input_text)
    
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text=response),
        model="llama3",
        stopReason="endTurn"
    )


# MCP client runner
async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, sampling_callback=handle_sampling_message) as session:
            await session.initialize()

            # Test interaction
            tools = await session.list_tools()
            print("Available tools:", tools)

            for p in tools:
                print("-", p)


if __name__ == "__main__":
    asyncio.run(run())