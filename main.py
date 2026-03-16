import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from scripts.dataset_generators.deepaction import main as download_deepaction_dataset
from scripts.dataset_generators.wanimate2_1 import main as download_wanimate_dataset
from scripts.dataset_generators.k400 import download_real_k400_videos
from scripts.cluster_videos import (
    VideoDirectoryDataset, 
    VideoResNet, 
    VideoResNet3D,
    ArcFaceLayer, 
    train_arcface,
    extract_embeddings,
    plot_clusters,
    evaluate_on_holdout_set,
    save_evaluation_results
)
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Downloading Datasets (Skipped for training run)")
    # download_deepaction_dataset()
    # download_wanimate_dataset()
    # download_real_k400_videos(target_count=300)

    DATA_DIR = "/workspace/video_data"
    EPOCHS = 10
    BATCH_SIZE = 8
    EMBEDDING_DIM = 512
    OUTPUT_DIR = "/workspace/video_cluster"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Setting up Model on {device}")

    # Note: For maximum anti-overfitting, will consider adding T.RandomHorizontalFlip() and T.ColorJitter() to a separate train_transform later
    transform = T.Compose([
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VideoDirectoryDataset(DATA_DIR, num_frames=16, transform=transform)
    
    if len(dataset) == 0:
        raise ValueError(f"No videos found in {DATA_DIR}! Check your directory structure.")
        
    num_classes = len(dataset.classes)
    logging.info(f"Found {len(dataset)} total videos across {num_classes} classes.")

    # Create a stratified train/test split of the dataset
    all_labels = [info[1] for info in dataset.videos]
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=0.2, 
        random_state=42, 
        stratify=all_labels
    )

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    logging.info(f"Dataset split: {len(train_subset)} training samples, {len(test_subset)} testing samples.")

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    train_eval_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    test_video_paths = [os.path.join(*dataset.videos[i][0].split(os.sep)[-2:]) for i in test_indices]

    # ==========================================
    # 2D RESNET PIPELINE
    # ==========================================
    logging.info("STARTING 2D MODEL PIPELINE")
    
    model_2d = VideoResNet(embedding_dim=EMBEDDING_DIM).to(device)
    arcface_2d = ArcFaceLayer(in_features=EMBEDDING_DIM, num_classes=num_classes).to(device)

    train_arcface(model_2d, arcface_2d, train_loader, epochs=EPOCHS, device=device, save_prefix="2d")

    logging.info("Extracting embeddings for evaluation...")
    train_emb_2d, train_lbl_2d = extract_embeddings(model_2d, train_eval_loader, device)
    test_emb_2d, test_lbl_2d = extract_embeddings(model_2d, test_loader, device)

    report_2d = evaluate_on_holdout_set(
        train_emb_2d, train_lbl_2d, test_emb_2d, test_lbl_2d, class_names=dataset.classes, n_neighbors=7
    )

    save_evaluation_results(report_2d, output_path=os.path.join(OUTPUT_DIR, "evaluation_metrics_2D.csv"))
    plot_clusters(test_emb_2d, test_lbl_2d, dataset.classes, video_paths=test_video_paths, output_path=os.path.join(OUTPUT_DIR, "test_set_cluster_plot_2D.png"))



    # ==========================================
    # 3D RESNET PIPELINE
    # ==========================================

    logging.info("STARTING 3D MODEL PIPELINE")
    
    model_3d = VideoResNet3D(embedding_dim=EMBEDDING_DIM).to(device)
    arcface_3d = ArcFaceLayer(in_features=EMBEDDING_DIM, num_classes=num_classes).to(device)
    train_arcface(model_3d, arcface_3d, train_loader, epochs=EPOCHS, device=device, save_prefix="3d")

    logging.info("Extracting embeddings for evaluation...")
    train_emb_3d, train_lbl_3d = extract_embeddings(model_3d, train_eval_loader, device)
    test_emb_3d, test_lbl_3d = extract_embeddings(model_3d, test_loader, device)

    report_3d = evaluate_on_holdout_set(
        train_emb_3d, train_lbl_3d, test_emb_3d, test_lbl_3d, class_names=dataset.classes, n_neighbors=7
    )

    save_evaluation_results(report_3d, output_path=os.path.join(OUTPUT_DIR, "evaluation_metrics_3D.csv"))
    plot_clusters(test_emb_3d, test_lbl_3d, dataset.classes, video_paths=test_video_paths, output_path=os.path.join(OUTPUT_DIR, "test_set_cluster_plot_3D.png"))

    logging.info("A/B Testing Pipeline Complete!")

if __name__ == "__main__":
    main()