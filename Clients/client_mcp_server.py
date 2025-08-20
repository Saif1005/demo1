from mcp import Server, Tool
from fetch_data import fetch_data
from clean_data import clean_data
from train_llava import train_llava
from download_model import download_model
from generate_profile import generate_profile
import os

def create_mcp_server(client_id: str):
    """Create an MCP server exposing client tools."""
    server = Server()

    # Register tools
    server.register_tool(Tool(
        name="download_model",
        description="Download LLaVA model for training",
        function=download_model
    ))

    server.register_tool(Tool(
        name="fetch_data",
        description="Fetch dummy data for training",
        function=fetch_data
    ))

    server.register_tool(Tool(
        name="clean_data",
        description="Clean the fetched data",
        function=lambda input_path: clean_data(input_path, "/data/cleaned_data.json")
    ))

    server.register_tool(Tool(
        name="train_llava",
        description="Fine-tune LLaVA model with LoRA",
        function=lambda client_id=client_id: train_llava("/data/cleaned_data.json", "/output", client_id)
    ))

    server.register_tool(Tool(
        name="generate_profile",
        description="Generate platform-specific profile",
        function=lambda client_id=client_id: generate_profile("/data/cleaned_data.json", "/model", "/output", client_id)
    ))

    return server

if __name__ == "__main__":
    client_id = os.environ.get("CLIENT_ID", "client1")
    server = create_mcp_server(client_id)
    server.run()