import flwr as fl
import torch
from client_workflow import run_client_workflow
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM
from peft import PeftModel

class LLaVAClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, model_name: str = "liuhaotian/llava-v1.5-7b"):
        self.client_id = client_id
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return model parameters as NumPy arrays."""
        model = PeftModel.from_pretrained(self.model, f"/output/lora_weights_{self.client_id}")
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters from NumPy arrays."""
        state_dict = PeftModel.from_pretrained(self.model, f"/output/lora_weights_{self.client_id}").state_dict()
        for i, (key, val) in enumerate(state_dict.items()):
            state_dict[key] = torch.from_numpy(parameters[i]).to(val.device)
        model = PeftModel.from_pretrained(self.model, f"/output/lora_weights_{self.client_id}")
        model.load_state_dict(state_dict)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model and return updated parameters."""
        self.set_parameters(parameters)
        # Run CrewAI workflow to fetch, clean, and train
        weights_path = run_client_workflow(self.client_id)
        # Load updated parameters
        params = self.get_parameters(config)
        return params, 2, {}  # 2 is dummy sample count

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the model (placeholder)."""
        self.set_parameters(parameters)
        return 0.0, 2, {"accuracy": 0.0}  # Dummy evaluation

def start_flower_client(client_id: str):
    """Start Flower client."""
    client = LLaVAClient(client_id)
    fl.client.start_numpy_client(server_address="mcp-host:8080", client=client)

if __name__ == "__main__":
    import sys
    start_flower_client(sys.argv[1] if len(sys.argv) > 1 else "client1")