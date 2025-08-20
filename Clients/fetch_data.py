import json
import os
from pathlib import Path

def fetch_data(input_path: str = "/data/raw_facebook_data.json", output_path: str = "/data/dummy_data.json"):
    """Fetch and preprocess unorganized Facebook data from PVC."""
    # Check if input file exists in PVC
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw data not found at {input_path}")

    # Load raw Facebook data
    with open(input_path, 'r') as f:
        raw_data = json.load(f)

    # Preprocess to extract relevant fields
    processed_data = []
    for item in raw_data:
        # Handle inconsistent data
        post_id = item.get("post_id", "unknown")
        text = item.get("message", item.get("text", ""))
        image_url = item.get("full_picture", item.get("image_url", ""))
        platform = item.get("platform", "facebook")

        # Skip entries with no text or image
        if not text or not image_url:
            continue

        processed_data.append({
            "post_id": post_id,
            "text": text,
            "image_url": image_url,
            "platform": platform
        })

    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2)

    return output_path

if __name__ == "__main__":
    fetch_data()