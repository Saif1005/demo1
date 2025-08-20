from kfp.v2 import dsl
from kfp.v2.dsl import component, pipeline, Output, Artifact

@component
def flower_client_op(client_id: str) -> str:
    import sys
    sys.path.append("/clients")
    from flower_client import start_flower_client
    start_flower_client(client_id)
    return f"/output/profile_{client_id}.json"

@component
def mcp_host_op(profile_paths: list) -> str:
    import sys
    sys.path.append("/host")
    from mcp_host import MCPHost
    mcp_host = MCPHost(output_dir="/output")
    return mcp_host.run_fl_rounds(profile_paths=profile_paths)

@pipeline(name="federated-llava-pipeline")
def fl_pipeline():
    client_ids = ["client1", "client2"]  # Example client IDs
    client_tasks = []
    for client_id in client_ids:
        task = flower_client_op(client_id=client_id)
        client_tasks.append(task.output)
    
    mcp_task = mcp_host_op(profile_paths=client_tasks)

if __name__ == "__main__":
    from kfp.v2.compiler import Compiler
    Compiler().compile(fl_pipeline, "fl_pipeline.yaml")