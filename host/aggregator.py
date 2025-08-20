import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
import os

def aggregate_models(client_weights_paths: list, output_dir: str):
    """Perform FedAvg aggregation of LoRA weights."""
    base_model_name = "liuhaotian/llava-v1.5-7b"
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)
    
    # Initialize aggregated weights
    aggregated_state_dict = None
    num_clients = len(client_weights_paths)
    
    for path in client_weights_paths:
        # Load client LoRA weights
        model = PeftModel.from_pretrained(base_model, path)
        state_dict = model.state_dict()
        
        if aggregated_state_dict is None:
            aggregated_state_dict = {k: torch.zeros_like(v) for k, v in state_dict.items()}
        
        # Accumulate weights
        for k, v in state_dict.items():
            aggregated_state_dict[k] += v / num_clients
    
    # Save aggregated model
    os.makedirs(output_dir, exist_ok=True)
    model = PeftModel.from_pretrained(base_model, client_weights_paths[0])  # Use first client's config
    model.load_state_dict(aggregated_state_dict)
    model.save_pretrained(f"{output_dir}/global_model")
    return f"{output_dir}/global_model"

if __name__ == "__main__":
    client_weights = ["output/lora_weights_client1", "output/lora_weights_client2"]
    aggregate_models(client_weights, "output/global")