import os
import shutil
import logging
from datasets import load_dataset, DownloadConfig, Video
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = "/workspace/video_data"
HF_CACHE_DIR = "/workspace/hf_cache"
NUM_VIDEOS_TO_DOWNLOAD = 300
OUTPUT_FOLDER_NAME = "wanimate2_1"


# ==========================================
# HELPER FUNCTION
# ==========================================
def extract_video_to_disk(video_obj, save_path):
    """Safely extracts video data from HF formats (bytes, paths, or strings) to an MP4."""
    if os.path.exists(save_path):
        logging.info(f"  File already exists, skipping: {os.path.basename(save_path)}")
        return True

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

# ==========================================
# DATASET DOWNLOADER
# ==========================================
def download_dataset(repo, ds_name, split, dl_config):
    """Downloads a specified number of videos from a given Hugging Face repository."""
    logging.info(f"--- Processing Dataset: {ds_name} ({repo}) ---")

    out_dir = os.path.join(BASE_DIR, OUTPUT_FOLDER_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # Check if enough files for this dataset already exist
    existing_files = [f for f in os.listdir(out_dir) if f.startswith(f"{ds_name}_") and f.endswith(".mp4")]
    if len(existing_files) >= NUM_VIDEOS_TO_DOWNLOAD:
        logging.info(f"  Skipping '{ds_name}': Found {len(existing_files)} files, meeting target of {NUM_VIDEOS_TO_DOWNLOAD}.")
        return

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
        dataset = dataset.cast_column("video", Video(decode=False))

    except Exception as e:
        logging.error(f"Failed to load {ds_name} from Hugging Face: {e}")
        return

    download_count = 0
    for i, row in enumerate(dataset):
        if download_count >= NUM_VIDEOS_TO_DOWNLOAD:
            break

        video_obj = row.get("video")
        if not video_obj: continue

        file_name = f"{ds_name}_{i:04d}.mp4"
        save_path = os.path.join(out_dir, file_name)

        if extract_video_to_disk(video_obj, save_path):
            download_count += 1
            logging.info(f"  Downloaded {download_count}/{NUM_VIDEOS_TO_DOWNLOAD} -> {file_name}")

    logging.info(f"Finished {ds_name}. Downloaded a total of {download_count} videos.")

# ==========================================
# MAIN ORCHESTRATION
# ==========================================
def main():
    """Sets up logging and orchestrates the download."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    dl_config = DownloadConfig(num_proc=4)

    datasets_to_download = {
        "wan_blockify_effect": "linoyts/wan_blockify_effect",
        "wan_putting_down_object": "linoyts/wan_putting_down_object",
        "wan_blinking": "linoyts/wan_blinking",
        "wan_shatter_effect": "linoyts/wan_shatter_effect",
        "wan_laughing": "linoyts/wan_laughing",
    }

    for name, repo in datasets_to_download.items():
        # Assuming all datasets have a 'train' split
        download_dataset(repo, name, "train", dl_config)

if __name__ == "__main__":
    main()