import os
import shutil
import logging
from datasets import load_dataset, DownloadConfig, Video
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = "/workspace/video_data"
HF_CACHE_DIR = "/workspace/hf_cache"
TOTAL_VIDEOS_TARGET = 400  # The absolute maximum for the whole folder!
OUTPUT_FOLDER_NAME = "wanimate2_1"

# ==========================================
# HELPER FUNCTION
# ==========================================
def extract_video_to_disk(video_obj, save_path):
    """Safely extracts video data from HF formats."""
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
        logging.error(f"  [!] Failed to save video: {e}")
    return False

# ==========================================
# DATASET DOWNLOADER
# ==========================================
def download_datasets(datasets_dict, dl_config):
    """Downloads videos from multiple repos until the folder hits the target count."""
    out_dir = os.path.join(BASE_DIR, OUTPUT_FOLDER_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Check the GLOBAL folder count before starting
    current_total = len([f for f in os.listdir(out_dir) if f.endswith(".mp4")])
    logging.info(f"Initial folder count: {current_total}/{TOTAL_VIDEOS_TARGET}")

    if current_total >= TOTAL_VIDEOS_TARGET:
        logging.info(f"Target of {TOTAL_VIDEOS_TARGET} videos already reached. Exiting.")
        return

    for ds_name, repo in datasets_dict.items():
        logging.info(f"--- Pulling from Repository: {ds_name} ({repo}) ---")

        try:
            # 2. No split parameter! Load everything safely.
            dataset_dict = load_dataset(
                repo,
                trust_remote_code=True,
                download_config=dl_config,
                token=os.environ.get("HF_TOKEN"),
                cache_dir=HF_CACHE_DIR
            )
        except Exception as e:
            logging.error(f"Failed to load {ds_name}: {e}")
            continue

        for split_name, dataset_split in dataset_dict.items():
            dataset_split = dataset_split.cast_column("video", Video(decode=False))
            
            for i, row in enumerate(dataset_split):
                # 3. Stop immediately if the folder is full
                if current_total >= TOTAL_VIDEOS_TARGET:
                    logging.info(f"Hit {TOTAL_VIDEOS_TARGET} capacity! Stopping all downloads.")
                    return

                video_obj = row.get("video")
                if not video_obj: continue

                file_name = f"{ds_name}_{i:04d}.mp4"
                save_path = os.path.join(out_dir, file_name)

                # 4. Only process and count if the file is genuinely new
                if not os.path.exists(save_path):
                    if extract_video_to_disk(video_obj, save_path):
                        current_total += 1
                        if current_total % 25 == 0:
                            logging.info(f"  Folder progress: {current_total}/{TOTAL_VIDEOS_TARGET}")

# ==========================================
# MAIN ORCHESTRATION
# ==========================================
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    dl_config = DownloadConfig(num_proc=4)

    datasets_to_download = {
        "wan_blockify_effect": "linoyts/wan_blockify_effect",
        "wan_putting_down_object": "linoyts/wan_putting_down_object",
        "wan_blinking": "linoyts/wan_blinking",
        "wan_shatter_effect": "linoyts/wan_shatter_effect",
        "wan_laughing": "linoyts/wan_laughing",
        "wan_putting_on_hat": "linoyts/wan_putting_on_hat",
        "wan_blowing_bubble_with_gum": "linoyts/wan_blowing_bubble_with_gum",
        "wan_scrolling_on_phone": "linoyts/wan_scrolling_on_phone",
        "wan_shrugging": "linoyts/wan_shrugging",
        "wan_shuffling_cards": "linoyts/wan_shuffling_cards",
        "wan_shaking_head": "linoyts/wan_shaking_head",
        "wan_buttoning_shirt": "linoyts/wan_buttoning_shirt",
        "wan_blowing_out_candle": "linoyts/wan_blowing_out_candle",
        "wan_popping_balloon": "linoyts/wan_popping_balloon",
        "wan_raising_eyebrows": "linoyts/wan_raising_eyebrows",
        "wan_licking_lips": "linoyts/wan_licking_lips",
        "wan_saluting": "linoyts/wan_saluting",
        "wan_bouncing_ball": "linoyts/wan_bouncing_ball",
        "wan_pouring_liquid": "linoyts/wan_pouring_liquid",
        "wan_crumble_disintegrate_effect": "linoyts/wan_crumble_disintegrate_effect",
        "wan_closing_umbrella": "linoyts/wan_closing_umbrella",
        "wan_showing_muscles": "linoyts/wan_showing_muscles",
        "wan_doing_single_squat": "linoyts/wan_doing_single_squat",
        "wan_rolling_eyes": "linoyts/wan_rolling_eyes",
        "wan_clapping_hands": "linoyts/wan_clapping_hands"
    }

    # Pass the whole dictionary at once so the folder capacity is tracked accurately
    download_datasets(datasets_to_download, dl_config)

if __name__ == "__main__":
    main()