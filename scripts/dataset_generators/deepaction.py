import os
import shutil
from collections import defaultdict
import logging
from datasets import load_dataset, DownloadConfig, Video
from dotenv import load_dotenv
load_dotenv()  

BASE_DIR = "/workspace/video_data"
HF_CACHE_DIR = "/workspace/hf_cache"
VIDEOS_PER_LABEL = 300

# ==========================================
# HELPER FUNCTIONS
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
        logging.error(f"  [!] Failed to save video: {e}")
        
    return False

def download_deepaction(dl_config):
    """Downloads and processes only the DeepAction dataset."""
    ds_name = "DeepAction"
    repo = "faridlab/deepaction_v1"
    split = "train"
    
    logging.info(f"--- Processing Dataset: {ds_name} ---")
    
    # 1. Load the dataset
    try:
        dataset = load_dataset(
            repo, 
            split=split,
            trust_remote_code=True,
            download_config=dl_config,
            token=os.environ.get("HF_TOKEN"),
            cache_dir=HF_CACHE_DIR
        )
        
        #Tell Hugging Face NOT to open the video files
        dataset = dataset.cast_column("video", Video(decode=False))
        
    except Exception as e:
        logging.error(f"Failed to load {ds_name} from Hugging Face: {e}")
        return

    logging.info(f"Using split: '{split}' ({len(dataset)} total items)")

    label_counts = defaultdict(int)
    checked_dirs = set()

    # 2. Iterate through the dataset
    for row in dataset:
        # DeepAction stores the generator as an integer label.
        label_id = row['label']
        model_name = dataset.features['label'].int2str(label_id)

        # In DeepAction, Pexels is the source for Real/Authentic videos
        if "pexels" in model_name.lower():
            target_label = "Real"
        else:
            target_label = model_name.replace("/", "_").replace(" ", "_")

        # Pre-scan directory on first encounter of a label to count existing files
        if target_label not in checked_dirs:
            out_dir = os.path.join(BASE_DIR, target_label)
            if os.path.isdir(out_dir):
                label_counts[target_label] = len([f for f in os.listdir(out_dir) if f.endswith(('.mp4', '.avi', '.mov'))])
            checked_dirs.add(target_label)

        # Skip if we already have enough videos for this class
        if label_counts.get(target_label, 0) >= VIDEOS_PER_LABEL:
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
            # Log progress
            if label_counts[target_label] % 10 == 0 or VIDEOS_PER_LABEL < 10:
                logging.info(f"  Downloaded {label_counts[target_label]}/{VIDEOS_PER_LABEL} for [{target_label}]")

    logging.info(f"Finished {ds_name}. Final counts:")
    for k, v in sorted(dict(label_counts).items()):
        logging.info(f"  - {k}: {v}")

# ==========================================
# MAIN ORCHESTRATION
# ==========================================
def main():
    """Sets up logging and orchestrates the download."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Use a custom download config to prevent timeouts
    dl_config = DownloadConfig(num_proc=4)
    download_deepaction(dl_config)

if __name__ == "__main__":
    main()