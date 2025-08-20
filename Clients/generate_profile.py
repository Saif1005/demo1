import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np

def generate_profile(data_path: str, model_dir: str, output_dir: str, client_id: str):
    """Generate a platform-specific profile using client LoRA weights."""
    # Load model and tokenizer from PVC
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, f"{output_dir}/lora_weights_{client_id}")

    # Load test data
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Generate embeddings for each data point
    model.eval()
    embeddings = []
    with torch.no_grad():
        for item in data[:10]:  # Limit to 10 samples for efficiency
            inputs = tokenizer(item["text"], return_tensors="pt", truncation=True, max_length=512).to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            # Use mean-pooled last hidden state as embedding
            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(embedding)

    # Average embeddings to create platform-specific profile
    profile = np.mean(embeddings, axis=0).tolist()

    # Save profile
    os.makedirs(output_dir, exist_ok=True)
    profile_path = f"{output_dir}/profile_{client_id}.json"
    with open(profile_path, 'w') as f:
        json.dump({"client_id": client_id, "profile": profile}, f, indent=2)

    return profile_path

if __name__ == "__main__":
    import sys
    generate_profile("/data/cleaned_data.json", "/model", "/output", sys.argv[1] if len(sys.argv) > 1 else "client1")