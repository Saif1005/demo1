import flwr as fl
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
import os
import numpy as np
from typing import Dict, List
from fuse_profiles import fuse_profiles

class MCPHost:
    def __init__(self, output_dir: str, num_rounds: int = 3):
        self.output_dir = output_dir
        self.num_rounds = num_rounds
        self.model_name = "liuhaotian/llava-v1.5-7b"
        self.client_profiles = {}  # Store client weights and profiles

    def get_initial_parameters(self):
        """Get initial model parameters."""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16
        )
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def aggregate(self, results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """Aggregate client weights using FedAvg."""
        aggregated_params = []
        num_examples_total = sum(num_examples for _, num_examples in results)
        for i in range(len(results[0][0])):
            weighted_sum = np.zeros_like(results[0][0][i])
            for params, num_examples in results:
                weighted_sum += params[i] * (num_examples / num_examples_total)
            aggregated_params.append(weighted_sum)
        return aggregated_params

    def save_global_model(self, parameters: List[np.ndarray]):
        """Save aggregated model."""
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16)
        state_dict = model.state_dict()
        for i, (key, val) in enumerate(state_dict.items()):
            state_dict[key] = torch.from_numpy(parameters[i]).to(val.device)
        model = PeftModel.from_pretrained(model, f"{self.output_dir}/lora_weights_client1")
        model.load_state_dict(state_dict)
        os.makedirs(self.output_dir, exist_ok=True)
        model.save_pretrained(f"{self.output_dir}/global")
        return f"{self.output_dir}/global"

    def fuse_client_profiles(self, profile_paths: List[str]):
        """Fuse platform-specific profiles into a general profile."""
        return fuse_profiles(profile_paths, self.output_dir)

    def run_fl_rounds(self, profile_paths: List[str] = None):
        """Run Flower server for FL rounds and fuse profiles."""
        strategy = fl.server.strategy.FedAvg(
            min_fit_clients=2,
            min_available_clients=2,
            on_fit_config_fn=lambda r: {},
            fit_metrics_aggregation_fn=None,
            evaluate_fn=None,
            initial_parameters=self.get_initial_parameters()
        )
        fl.server.start_server(
            server_address="[::]:8080",
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy
        )
        # Fuse profiles after FL rounds
        if profile_paths:
            return self.fuse_client_profiles(profile_paths)
        return f"{self.output_dir}/global"

if __name__ == "__main__":
    mcp_host = MCPHost(output_dir="/output")
    mcp_host.run_fl_rounds()