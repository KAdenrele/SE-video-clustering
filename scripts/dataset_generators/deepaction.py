import os
import shutil
from collections import defaultdict
import logging
from datasets import load_dataset, DownloadConfig, Video
from dotenv import load_dotenv

load_dotenv()  

BASE_DIR = "/workspace/video_data"
HF_CACHE_DIR = "/workspace/hf_cache"
VIDEOS_PER_LABEL = 400

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def extract_video_to_disk(video_obj, save_path):
    """Safely extracts video data from HF formats (bytes, paths, or strings) to an MP4."""
    if os.path.exists(save_path):
        return True # File already exists, skip downloading
        
    try:
        if isinstance(video_obj, dict) and "bytes" in video_obj and video_obj["bytes"]:
            with open(save_path, "wb") as f:
                f.write(video_obj["bytes"])
            return True
            
        elif isinstance(video_obj, dict) and "path" in video_obj and os.path.exists(video_obj["path"]):
            shutil.copy(video_obj["path"], save_path)
            return True
            
        elif isinstance(video_obj, str) and os.path.exists(video_obj):
            shutil.copy(video_obj, save_path)
            return True
            
    except Exception as e:
        logging.error(f"  [!] Exception while saving video: {e}")
        return False
        
    # If we reach here, the HF cache path is broken or missing
    logging.debug(f"  [!] Skipped row: Video file missing from HF cache.")
    return False

def download_deepaction(dl_config):
    """Downloads and processes ALL splits of the DeepAction dataset."""
    ds_name = "DeepAction"
    repo = "faridlab/deepaction_v1"
    
    logging.info(f"--- Processing Dataset: {ds_name} ---")
    
    try:
        dataset_dict = load_dataset(
            repo, 
            trust_remote_code=True,
            download_config=dl_config,
            token=os.environ.get("HF_TOKEN"),
            cache_dir=HF_CACHE_DIR,
            #Force HF to re-download and fix the broken cache
            download_mode="force_redownload" 
        )
    except Exception as e:
        logging.error(f"Failed to load {ds_name} from Hugging Face: {e}")
        return

    label_counts = defaultdict(int)
    initial_counts = defaultdict(int) # Tracks how many we started with to prevent duplicates
    rows_seen = defaultdict(int)      # Tracks our position in the dataset
    checked_dirs = set()

    for split_name, dataset_split in dataset_dict.items():
        logging.info(f"Scanning split: '{split_name}' ({len(dataset_split)} items)")
        
        dataset_split = dataset_split.cast_column("video", Video(decode=False))

        for row in dataset_split:
            label_id = row['label']
            model_name = dataset_split.features['label'].int2str(label_id)

            if "pexels" in model_name.lower():
                target_label = "Real"
            else:
                target_label = model_name.replace("/", "_").replace(" ", "_")

            if target_label not in checked_dirs:
                out_dir = os.path.join(BASE_DIR, target_label)
                if os.path.isdir(out_dir):
                    existing = len([f for f in os.listdir(out_dir) if f.endswith(('.mp4', '.avi', '.mov'))])
                    label_counts[target_label] = existing
                    initial_counts[target_label] = existing
                checked_dirs.add(target_label)

            # Track that we are looking at a new row for this class
            rows_seen[target_label] += 1

            # Check if we hit the total cap (400)
            if label_counts.get(target_label, 0) >= VIDEOS_PER_LABEL:
                continue

            # Skip rows that were already downloaded in previous runs
            if rows_seen[target_label] <= initial_counts.get(target_label, 0):
                continue

            out_dir = os.path.join(BASE_DIR, target_label)
            os.makedirs(out_dir, exist_ok=True)

            video_obj = row.get("video")
            if not video_obj:
                continue

            file_name = f"{ds_name}_{target_label}_{label_counts[target_label]:03d}.mp4"
            save_path = os.path.join(out_dir, file_name)

            if extract_video_to_disk(video_obj, save_path):
                label_counts[target_label] += 1
                if label_counts[target_label] % 10 == 0: # Log every 10 for better visibility
                    logging.info(f"  Downloaded {label_counts[target_label]}/{VIDEOS_PER_LABEL} for [{target_label}]")

    logging.info(f"Finished {ds_name}. Final counts:")
    for k, v in sorted(dict(label_counts).items()):
        logging.info(f"  - {k}: {v}")

# ==========================================
# MAIN ORCHESTRATION
# ==========================================
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    dl_config = DownloadConfig(num_proc=4)
    download_deepaction(dl_config)

if __name__ == "__main__":
    main()