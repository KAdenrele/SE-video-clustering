import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Import from your modules
from scripts.dataset_downloader import main as download_datasets
from scripts.cluster_videos import (
    VideoDirectoryDataset, 
    VideoResNet101, 
    ArcFaceLayer, 
    train_arcface, 
    extract_embeddings, 
    plot_clusters_plotly
)

def main():
    # 1. Download Datasets
    print("--- Step 1: Downloading Datasets ---")
    download_datasets()

    # 2. Configuration
    # Using the Docker container's mapped path
    DATA_DIR = "/workspace/video_data" 
    EPOCHS = 10
    BATCH_SIZE = 8
    EMBEDDING_DIM = 512
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Step 2: Setting up Model on {device} ---")

    # 3. Setup Data
    transform = T.Compose([
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # We need to ensure the dataset looks at the right depth. 
    # (See note below on how to fix the downloader script)
    dataset = VideoDirectoryDataset(DATA_DIR, num_frames=16, transform=transform)
    
    if len(dataset) == 0:
        raise ValueError(f"No videos found in {DATA_DIR}! Check your directory structure.")
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    num_classes = len(dataset.classes)
    print(f"Found {len(dataset)} videos across {num_classes} classes.")

    # 4. Initialize Model & ArcFace
    model = VideoResNet101(embedding_dim=EMBEDDING_DIM).to(device)
    arcface_layer = ArcFaceLayer(in_features=EMBEDDING_DIM, num_classes=num_classes).to(device)

    # 5. Train
    print("--- Step 3: Training ArcFace ---")
    train_arcface(model, arcface_layer, dataloader, epochs=EPOCHS, device=device)

    # 6. Extract
    print("--- Step 4: Extracting Embeddings ---")
    dataloader_eval = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    embeddings, labels = extract_embeddings(model, dataloader_eval, device)

    # 7. Visualize
    print("--- Step 5: Visualizing Clusters ---")
    # Ensure the output directory exists so matplotlib doesn't throw a FileNotFoundError
    output_dir = "/workspace/video_cluster"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_clusters(
        embeddings, 
        labels, 
        dataset.classes, 
        output_path=os.path.join(output_dir, "cluster_output.png")
    )
    print("Pipeline Complete!")

if __name__ == "__main__":
    main()