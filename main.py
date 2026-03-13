import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from scripts.dataset_generators.deepaction import main as download_deepaction_dataset
from scripts.dataset_generators.wanimate2_1 import main as download_wanimate_dataset
from scripts.cluster_videos import (
    VideoDirectoryDataset, 
    VideoResNet101, 
    ArcFaceLayer, 
    train_arcface, 
    extract_embeddings, 
    plot_clusters
)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Downloading Datasets")
    download_deepaction_dataset()
    download_wanimate_dataset()

    DATA_DIR = "/workspace/video_data"
    EPOCHS = 10
    BATCH_SIZE = 8
    EMBEDDING_DIM = 512
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Setting up Model on {device}")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VideoDirectoryDataset(DATA_DIR, num_frames=16, transform=transform)
    
    if len(dataset) == 0:
        raise ValueError(f"No videos found in {DATA_DIR}! Check your directory structure.")
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    num_classes = len(dataset.classes)
    logging.info(f"Found {len(dataset)} videos across {num_classes} classes.")


    model = VideoResNet101(embedding_dim=EMBEDDING_DIM).to(device)
    arcface_layer = ArcFaceLayer(in_features=EMBEDDING_DIM, num_classes=num_classes).to(device)

    logging.info("Training ArcFace")
    train_arcface(model, arcface_layer, dataloader, epochs=EPOCHS, device=device)


    logging.info("Extracting Embeddings")
    dataloader_eval = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    embeddings, labels = extract_embeddings(model, dataloader_eval, device)


    logging.info("Visualizing Clusters")
    output_dir = "/workspace/video_cluster"
    os.makedirs(output_dir, exist_ok=True)

    # Grab the video paths in the exact same order as the extracted embeddings
    # We slice it to only show "Label/Video.mp4" so the hover box isn't cluttered
    video_paths = [os.path.join(*vid_info[0].split(os.sep)[-2:]) for vid_info in dataset.videos]
    
    plot_clusters(
        embeddings, 
        labels, 
        dataset.classes, 
        video_paths=video_paths,
        output_path=os.path.join(output_dir, "cluster_output.png")
    )
    logging.info("Pipeline Complete!")

if __name__ == "__main__":
    main()