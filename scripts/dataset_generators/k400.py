import os
import shutil
import tarfile
import urllib.request
import logging

def download_real_k400_videos(target_count=300):
    """
    Downloads Kinetics-400 validation tarballs, extracts exactly `target_count` 
    videos into a clean folder, and removes all temporary files.
    """
    # The clean target directory for your PoC
    TARGET_DIR = "/workspace/video_data/Real"
    
    # K400 provides a text file containing the URLs to all the .tar.gz archives
    K400_URL_LIST = "https://s3.amazonaws.com/kinetics/400/val/k400_val_path.txt"
    
    # Ensure our perfectly clean target directory exists
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # 1. Check how many videos we already have
    existing_videos = [f for f in os.listdir(TARGET_DIR) if f.endswith(('.mp4', '.avi', '.mkv'))]
    current_count = len(existing_videos)
    
    logging.info(f"Initial 'Real' folder count: {current_count}/{target_count}")
    if current_count >= target_count:
        logging.info(f"Target of {target_count} real videos already met. Exiting.")
        return

    # 2. Fetch the list of tarball URLs
    logging.info("Fetching K400 archive list...")
    try:
        response = urllib.request.urlopen(K400_URL_LIST)
        urls = response.read().decode('utf-8').strip().split('\n')
    except Exception as e:
        logging.error(f"Failed to fetch K400 URL list: {e}")
        return

    # 3. Stream and extract archives one by one
    for url in urls:
        if not url.strip():
            continue
            
        if current_count >= target_count:
            break # Stop downloading entirely!

        filename = os.path.basename(url)
        temp_tar_path = os.path.join(TARGET_DIR, filename)

        logging.info(f"Downloading archive: {filename}...")
        try:
            urllib.request.urlretrieve(url, temp_tar_path)
        except Exception as e:
            logging.warning(f"Failed to download {url}: {e}. Skipping to next.")
            continue

        logging.info(f"Extracting videos from {filename}...")
        try:
            # 4. Surgically extract ONLY the video files directly into the root folder
            with tarfile.open(temp_tar_path, "r:gz") as tar:
                for member in tar.getmembers():
                    # Stop extracting if we hit the limit
                    if current_count >= target_count:
                        break
                        
                    # Check if it's a video file (ignoring directories and metadata)
                    if member.isfile() and member.name.lower().endswith(('.mp4', '.avi', '.mkv')):
                        # Flatten the path: discard any internal folders from the tarball
                        clean_filename = f"k400_{os.path.basename(member.name)}"
                        dest_path = os.path.join(TARGET_DIR, clean_filename)
                        
                        if not os.path.exists(dest_path):
                            # Extract the raw file object and write it directly to our clean folder
                            with tar.extractfile(member) as source, open(dest_path, "wb") as target:
                                shutil.copyfileobj(source, target)
                            
                            current_count += 1
                            if current_count % 50 == 0:
                                logging.info(f"  Folder progress: {current_count}/{target_count}")
                                
        except Exception as e:
            logging.error(f"Error extracting {filename}: {e}")
            
        # 5. Destroy the massive tarball to keep the hard drive clean
        logging.info(f"Cleaning up temporary archive: {filename}")
        if os.path.exists(temp_tar_path):
            os.remove(temp_tar_path)

    logging.info(f"Finished. Total 'Real' videos: {current_count}")

# Execution block for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # You can easily change this parameter to 400 (or anything else) when you call it in main.py
    download_real_k400_videos(target_count=400)