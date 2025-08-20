
import torch
import json
import os
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
import requests
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms

def train_llava(data_path: str, output_dir: str, client_id: str, model_dir: str = "/model"):
    """Fine-tune LLaVA 1.5 (7B) with LoRA on text and image data from PVC."""
    # Load model and processor from PVC
    processor = AutoProcessor.from_pretrained(model_dir)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_dir, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Applicable aux couches du transformer
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"  # Compatible avec LLaVA
    )
    model = get_peft_model(model, lora_config)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensionner pour CLIP-ViT
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Dummy training loop (simplified for demo)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for item in data:
        try:
            # Préparer le texte
            text = item["text"]
            
            # Télécharger et prétraiter l'image depuis image_url
            if "image_url" in item and item["image_url"]:
                response = requests.get(item["image_url"])
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image_tensor = transform(image).to(model.device)
            else:
                # Si pas d'image, utiliser un placeholder ou sauter
                image_tensor = None
            
            # Préparer les entrées pour LLaVA
            inputs = processor(
                text=text,
                images=image_tensor,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)
            
            # Entraînement
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except Exception as e:
            print(f"Erreur lors du traitement de l'élément {item}: {e}")
            continue
    
    # Save LoRA weights
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(f"{output_dir}/lora_weights_{client_id}")
    return f"{output_dir}/lora_weights_{client_id}"

if __name__ == "__main__":
    import sys
    train_llava("/data/cleaned_data.json", "/output", sys.argv[1] if len(sys.argv) > 1 else "client1")
