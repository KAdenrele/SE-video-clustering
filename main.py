import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from scripts.dataset_generators.deepaction import main as download_deepaction_dataset
from scripts.dataset_generators.wanimate2_1 import main as download_wanimate_dataset
from scripts.cluster_videos import (
    VideoDirectoryDataset, 
    VideoResNet, 
    ArcFaceLayer, 
    train_arcface,
    extract_embeddings,
    plot_clusters,
    evaluate_with_knn_kfold,
    evaluate_on_holdout_set,
    save_evaluation_results
)
from sklearn.model_selection import train_test_split
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

    # Create DataLoaders for training and evaluation
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = VideoResNet(embedding_dim=EMBEDDING_DIM).to(device)
    arcface_layer = ArcFaceLayer(in_features=EMBEDDING_DIM, num_classes=num_classes).to(device)

    logging.info("Training ArcFace on the training set")
    train_arcface(model, arcface_layer, train_loader, epochs=EPOCHS, device=device)

    # Extract embeddings for both train and test sets for evaluation
    logging.info("Extracting embeddings for training set...")
    train_eval_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    train_embeddings, train_labels = extract_embeddings(model, train_eval_loader, device)

    logging.info("Extracting embeddings for test set...")
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device)

    #Evaluation
    output_dir = "/workspace/video_cluster"
    os.makedirs(output_dir, exist_ok=True)

    #K-Fold CV on the training data to assess embedding space quality
    kfold_report_df = evaluate_with_knn_kfold(
        train_embeddings, 
        train_labels, 
        class_names=dataset.classes, 
        n_splits=5, 
        n_neighbors=7
    )

    #Evaluation on the holdout test set to assess generalization
    holdout_report_df = evaluate_on_holdout_set(
        train_embeddings, train_labels, 
        test_embeddings, test_labels, 
        class_names=dataset.classes, 
        n_neighbors=7
    )

    #Save both reports to a single CSV file
    save_evaluation_results(
        kfold_report_df, 
        holdout_report_df, 
        output_path=os.path.join(output_dir, "evaluation_metrics.csv")
    )

    logging.info("Visualising Test Set Clusters")
    test_video_paths = [os.path.join(*dataset.videos[i][0].split(os.sep)[-2:]) for i in test_indices]
    
    plot_clusters(
        test_embeddings, 
        test_labels, 
        dataset.classes, 
        video_paths=test_video_paths,
        output_path=os.path.join(output_dir, "test_set_cluster_plot.png")
    )
    logging.info("Pipeline Complete!")

if __name__ == "__main__":
    main()