import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from scripts.dataset_generators.deepaction import main as download_deepaction_dataset
from scripts.dataset_generators.wanimate2_1 import main as download_wanimate_dataset
from scripts.dataset_generators.k400 import download_real_k400_videos
from scripts.media_processes.video_processes_pipeline import process_videos
from scripts.cluster_videos import (
    VideoDirectoryDataset, 
    VideoResNet, 
    VideoResNet3D,
    ArcFaceLayer, 
    train_arcface,
    extract_embeddings,
    plot_clusters,
    evaluate_on_holdout_set,
    generate_report_from_results,
    evaluate_on_transformed_data,
    calculate_cluster_distances,
    save_evaluation_results
)
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    DATA_DIR = "/workspace/video_data"
    EPOCHS = 20
    BATCH_SIZE = 8
    EMBEDDING_DIM = 512
    OUTPUT_DIR = "/workspace/video_cluster/outputs"
    TRANSFORMED_DATA_DIR = "/workspace/video_cluster/transformed_data"
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
    
    test_videos = [dataset.videos[i] for i in test_indices]
    test_video_paths = [os.path.join(os.path.basename(os.path.dirname(p)), os.path.basename(p)) for p, l in test_videos]

    # ==========================================
    # 2D RESNET PIPELINE
    # ==========================================
    # logging.info("STARTING 2D MODEL PIPELINE")
    
    # model_2d = VideoResNet(embedding_dim=EMBEDDING_DIM).to(device)
    # arcface_2d = ArcFaceLayer(in_features=EMBEDDING_DIM, num_classes=num_classes).to(device)

    # train_arcface(model_2d, arcface_2d, train_loader, epochs=EPOCHS, device=device, save_prefix="2d")

    # logging.info("Extracting embeddings for evaluation...")
    # train_emb_2d, train_lbl_2d = extract_embeddings(model_2d, train_eval_loader, device)
    # test_emb_2d, test_lbl_2d = extract_embeddings(model_2d, test_loader, device)

    # report_2d, _ = evaluate_on_holdout_set(
    #     train_emb_2d, train_lbl_2d, test_emb_2d, test_lbl_2d, class_names=dataset.classes, n_neighbors=7
    # )

    # save_evaluation_results(report_2d, output_path=os.path.join(OUTPUT_DIR, "evaluation_metrics_2D.csv"))
    # plot_clusters(test_emb_2d, test_lbl_2d, dataset.classes, video_paths=test_video_paths, output_path=os.path.join(OUTPUT_DIR, "test_set_cluster_plot_2D.png"))



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

    report_3d, knn_3d = evaluate_on_holdout_set(
        train_emb_3d, train_lbl_3d, test_emb_3d, test_lbl_3d, class_names=dataset.classes, n_neighbors=7
    )

    save_evaluation_results(report_3d, output_path=os.path.join(OUTPUT_DIR, "evaluation_metrics_3D.csv"))
    plot_clusters(test_emb_3d, test_lbl_3d, dataset.classes, video_paths=test_video_paths, output_path=os.path.join(OUTPUT_DIR, "test_set_cluster_plot_3D.png"))
    calculate_cluster_distances(test_emb_3d, test_lbl_3d, dataset.classes, output_path=os.path.join(OUTPUT_DIR, "cluster_distances_3D.csv"))

    # =================================================
    # 3D RESNET - TRANSFORMED HOLDOUT SET EVALUATION
    # =================================================
    logging.info("STARTING EVALUATION ON TRANSFORMED HOLDOUT SET (3D MODEL)")

    if os.path.isdir(TRANSFORMED_DATA_DIR):
        transformed_results_df, transformed_embeddings, transformed_labels = evaluate_on_transformed_data(
            knn_classifier=knn_3d,
            model=model_3d,
            test_videos=test_videos,
            transformed_data_dir=TRANSFORMED_DATA_DIR,
            dataset=dataset,
            device=device,
            transform=transform
        )
        calculate_cluster_distances(transformed_embeddings, transformed_labels, dataset.classes, output_path=os.path.join(OUTPUT_DIR, "cluster_distances_transformed_3D.csv"))

        # Save the detailed, per-video results
        transformed_results_path = os.path.join(OUTPUT_DIR, "transformed_evaluation_raw_results_3D.csv")
        transformed_results_df.to_csv(transformed_results_path, index=False)
        logging.info(f"Saved raw transformed evaluation results to {transformed_results_path}")

        # Generate, print, and save the summary classification report
        report_df = generate_report_from_results(transformed_results_df, dataset.classes)
        
        logging.info("Classification Report for Transformed Holdout Set (3D Model)")
        print(report_df)

        save_evaluation_results(report_df, output_path=os.path.join(OUTPUT_DIR, "transformed_evaluation_metrics_3D.csv"), header="Transformed Holdout Set Classification Report (3D Model)")

        logging.info("Generating cluster plot for transformed data...")
        plot_clusters(
            transformed_embeddings,
            transformed_labels,
            dataset.classes,
            output_path=os.path.join(OUTPUT_DIR, "transformed_test_set_cluster_plot_3D.png")
        )
    else:
        logging.warning(f"Transformed data directory not found at {TRANSFORMED_DATA_DIR}. Skipping evaluation on transformed videos.")

    logging.info("A/B Testing Pipeline Complete!")

if __name__ == "__main__":
    #logging.info("Downloading Datasets")
    # download_deepaction_dataset()
    # download_wanimate_dataset()
    # download_real_k400_videos(target_count=300)
    #logging.info("Transforming Videos with Social Media Simulators")
    #process_videos(input_dir="/workspace/video_data", output_dir="/workspace/video_cluster/transformed_data")
    logging.info("Running full training and evaluation pipeline on original and transformed videos")
    main()