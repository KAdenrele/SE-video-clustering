import os
import subprocess
from scripts.media_processes.media_processes import SocialMediaSimulator
from tqdm import tqdm

def process_videos(input_dir, output_dir):
    """
    Processes all videos in the input directory through various social media pipelines
    and saves the results in a structured output directory.
    """
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {root}", unit="file"):
            if file.endswith((".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png", ".heic", ".webp", ".tiff")):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                dir_name = os.path.dirname(relative_path)

                structured_output_dir = os.path.join(output_dir, dir_name)
                simulator = SocialMediaSimulator(base_output_dir=structured_output_dir)
                
                # The simulator now handles creating the correct subdirectories like
                # 'facebook', 'instagram_feed', 'whatsapp_high', etc.
                simulator.facebook(input_path)
                simulator.instagram(input_path, post_type='feed')
                simulator.instagram(input_path, post_type='story')
                simulator.whatsapp(input_path, quality_mode='standard', upload_type='media')
                simulator.whatsapp(input_path, quality_mode='high', upload_type='media')
                simulator.signal(input_path, quality_setting='standard', as_document=False)
                simulator.signal(input_path, quality_setting='standard', as_document=True)
                simulator.telegram(input_path, as_document=False)
                simulator.telegram(input_path, as_document=True)
                simulator.tiktok(input_path)

