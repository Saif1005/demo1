import json
import re
import os

def clean_data(input_path: str, output_path: str):
    """Clean unorganized Facebook data for LLaVA fine-tuning."""
    # Load input data
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    cleaned_data = []
    for item in data:
        # Ensure required fields exist
        if not isinstance(item, dict) or "text" not in item or "image_url" not in item:
            continue
        
        text = item["text"].strip()
        image_url = item["image_url"]
        
        # Skip empty or very short text
        if not text or len(text) < 10:  # Stricter threshold for social media
            continue
        
        # Remove posts with only non-alphanumeric content (e.g., emojis)
        if not re.search(r'[a-zA-Z0-9]', text):
            continue
        
        # Clean text: remove URLs, excessive whitespace, and specific social media artifacts
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'#[^\s]+', '', text)  # Remove hashtags
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Skip if text is empty after cleaning
        if not text:
            continue
        
        # Validate image URL (basic check)
        if not (image_url.startswith("http://") or image_url.startswith("https://")):
            continue

        # Format for LLaVA
        cleaned_data.append({
            "image": image_url,
            "text": text
        })
    
    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    return output_path

if __name__ == "__main__":
    clean_data("/data/dummy_data.json", "/data/cleaned_data.json")