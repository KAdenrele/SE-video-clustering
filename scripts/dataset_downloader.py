import os
import shutil
from collections import defaultdict
from datasets import load_dataset, DownloadConfig


BASE_DIR = "/workspace/video_data"
VIDEOS_PER_LABEL = 50

# Hugging Face schemas vary. We map the expected columns for each dataset here.
DATASET_CONFIGS = {
    "AEGIS": {
        "repo": "Clarifiedfish/AEGIS",
        "split_preference": ["test", "train"], 
        "video_col": "video", 
        "label_col": "label",       # Usually 'real' or 'fake'
        "model_col": "generator",   # The specific model (e.g., Sora, Kling)
        "trust_remote": False
    },
    "DeepAction": {
        "repo": "faridlab/deepaction_v1",
        "split_preference": ["train"], # DeepAction only provides a train split
        "video_col": "video",
        "label_col": "label", 
        "model_col": "model", 
        "trust_remote": True        # Required by the authors' custom loading script
    },
    "SynthVidDetect": {
        "repo": "ductai199x/synth-vid-detect",
        "split_preference": ["test", "train"],
        "video_col": "video",
        "label_col": "label",
        "model_col": "source",      # Uses 'source' to denote the generator
        "trust_remote": False
    },
    "FakeParts": {
        "repo": "hi-paris/FakeParts_Legacy",
        "split_preference": ["test", "train"],
        "video_col": "video",
        "label_col": "label",
        "model_col": "generator",
        "trust_remote": False
    }
}

# ==========================================
#HELPER FUNCTIONS
# ==========================================
def extract_video_to_disk(video_obj, save_path):
    """Safely extracts video data from HF formats (bytes, paths, or strings) to an MP4."""
    if os.path.exists(save_path):
        return True # File already exists, skip downloading
        
    try:
        # Format 1: HF Dictionary with raw bytes
        if isinstance(video_obj, dict) and "bytes" in video_obj and video_obj["bytes"]:
            with open(save_path, "wb") as f:
                f.write(video_obj["bytes"])
            return True
            
        # Format 2: HF Dictionary with a local cached path
        elif isinstance(video_obj, dict) and "path" in video_obj and os.path.exists(video_obj["path"]):
            shutil.copy(video_obj["path"], save_path)
            return True
            
        # Format 3: Direct string path to a cached file
        elif isinstance(video_obj, str) and os.path.exists(video_obj):
            shutil.copy(video_obj, save_path)
            return True
            
    except Exception as e:
        print(f"  [!] Failed to save video: {e}")
        
    return False

# ==========================================
# 3. MAIN ORCHESTRATION
# ==========================================
def main():
    # Use a custom download config to prevent timeouts on massive video datasets
    dl_config = DownloadConfig(num_proc=4)

    for ds_name, config in DATASET_CONFIGS.items():
        print(f"\n{'='*40}\nProcessing Dataset: {ds_name}\n{'='*40}")
        
        # 1. Load the dataset
        try:
            dataset_dict = load_dataset(
                config["repo"], 
                trust_remote_code=config["trust_remote"],
                download_config=dl_config
            )
        except Exception as e:
            print(f"Failed to load {ds_name} from Hugging Face: {e}")
            continue
            
        # 2. Find the right split (prefer 'test', fallback to 'train')
        split_to_use = None
        for split in config["split_preference"]:
            if split in dataset_dict:
                split_to_use = split
                break
                
        if not split_to_use:
            print(f"Could not find valid split for {ds_name}. Available: {dataset_dict.keys()}")
            continue
            
        dataset = dataset_dict[split_to_use]
        print(f"Using split: '{split_to_use}' ({len(dataset)} total items)")

        # Tracker for how many videos we have per label
        label_counts = defaultdict(int)

        # 3. Iterate through the dataset
        for i, row in enumerate(dataset):
            # Extract metadata safely using .get() in case a row is missing data
            is_real = str(row.get(config["label_col"], "")).strip().lower() in ["real", "authentic", "true", "1"]
            model_name = str(row.get(config["model_col"], "Unknown")).strip()
            
            # Determine the folder name
            if is_real or "pexels" in model_name.lower() or "videoasid" in model_name.lower():
                target_label = "Real"
            else:
                target_label = model_name.replace("/", "_").replace(" ", "_")
                
            # Check if we already have enough videos for this specific label
            if label_counts[target_label] >= VIDEOS_PER_LABEL:
                # If we have 50 of EVERYTHING, we can optimize and break early
                # (Commented out because HF datasets can have dozens of models, 
                # but you can add logic here to break if len(label_counts) == expected_models)
                continue
                
            # 4. Setup Directories
            out_dir = os.path.join(BASE_DIR, target_label)
            os.makedirs(out_dir, exist_ok=True)
            
            # 5. Extract and Save
            video_obj = row.get(config["video_col"])
            if not video_obj:
                continue
                
            file_name = f"{ds_name}_{target_label}_{label_counts[target_label]:03d}.mp4"
            save_path = os.path.join(out_dir, file_name)
            
            success = extract_video_to_disk(video_obj, save_path)
            
            if success:
                label_counts[target_label] += 1
                if label_counts[target_label] % 10 == 0: # Print progress every 10 videos
                    print(f"  Downloaded {label_counts[target_label]}/{VIDEOS_PER_LABEL} for [{target_label}]")

        print(f"Finished {ds_name}. Final counts:")
        for k, v in dict(label_counts).items():
            print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()