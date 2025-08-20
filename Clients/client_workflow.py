from crewai import Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
import os

def run_client_workflow(client_id: str):
    """Run CrewAI workflow for MCP client tools, including model download."""
    # Define MCP server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "clients.client_mcp_server"],
        env={"UV_PYTHON": "3.12", "CLIENT_ID": client_id, **os.environ}
    )

    # Initialize MCP server adapter
    mcp_adapter = MCPServerAdapter(server_params)

    # Define agent
    agent = Agent(
        role=f"MCP Client Agent ({client_id})",
        goal=f"Execute federated learning workflow to produce LoRA weights for client {client_id}",
        backstory=f"An agent managing local FL tasks for client {client_id} on its platform.",
        tools=[mcp_adapter],
        verbose=True
    )

    # Define tasks using MCP server tools
    download_task = Task(
        description="Download LLaVA model for training",
        agent=agent,
        expected_output="Path to downloaded model",
        tool="download_model",
        inputs={}
    )

    fetch_task = Task(
        description="Fetch dummy data for training",
        agent=agent,
        expected_output="Path to fetched data JSON file",
        tool="fetch_data",
        inputs={}
    )

    clean_task = Task(
        description="Clean the fetched data",
        agent=agent,
        expected_output="Path to cleaned data JSON file",
        tool="clean_data",
        inputs={"input_path": fetch_task.output}
    )

    train_task = Task(
        description="Fine-tune LLaVA model with LoRA",
        agent=agent,
        expected_output=f"Path to LoRA weights for client {client_id}",
        tool="train_llava",
        inputs={"client_id": client_id}
    )

    # Create CrewAI workflow
    crew = Crew(
        agents=[agent],
        tasks=[download_task, fetch_task, clean_task, train_task],
        process=Process.sequential,
        verbose=True
    )

    # Execute workflow and return weights path
    with mcp_adapter:
        result = crew.kickoff()
    return result

if __name__ == "__main__":
    import sys
    run_client_workflow(sys.argv[1] if len(sys.argv) > 1 else "client1")