import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def download_model(model_name: str = "liuhaotian/llava-v1.5-7b", output_dir: str = "/model"):
    """Download LLaVA model and tokenizer from Hugging Face."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Download model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=output_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=output_dir)
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return output_dir

if __name__ == "__main__":
    download_model()