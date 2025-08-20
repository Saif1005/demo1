import json
import os
import numpy as np
from typing import List

def fuse_profiles(profile_paths: List[str], output_dir: str):
    """Fuse platform-specific profiles into a general profile."""
    profiles = []
    for path in profile_paths:
        with open(path, 'r') as f:
            profile_data = json.load(f)
            profiles.append(np.array(profile_data["profile"]))

    # Average profiles to create general profile
    general_profile = np.mean(profiles, axis=0).tolist()

    # Save general profile
    os.makedirs(output_dir, exist_ok=True)
    general_profile_path = f"{output_dir}/general_profile.json"
    with open(general_profile_path, 'w') as f:
        json.dump({"general_profile": general_profile}, f, indent=2)

    return general_profile_path

if __name__ == "__main__":
    fuse_profiles(["/output/profile_client1.json", "/output/profile_client2.json"], "/output")