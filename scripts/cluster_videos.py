import os
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from scipy.spatial.distance import cdist
import logging

LABEL_MAP = {
    "BDAnimateDiffLightning": "ByteDance AnimateDiff Lightning",
    "CogVideoX5B": "CogVideoX",
    "LTX2_3": "LTX Video",
    "Real": "Authentic (Real Video)",
    "RunwayML": "Runway",
    "StableDiffusion": "Stable Diffusion",
    "VideoPoet": "Google VideoPoet",
    "Wanimate": "Wanimate",
}


class VideoDirectoryDataset(Dataset):
    """Loads videos from a directory where subfolders represent class labels."""
    def __init__(self, root_dir, num_frames=16, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.videos = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir): continue
            for vid_file in os.listdir(cls_dir):
                if vid_file.endswith(('.mp4', '.avi', '.mov')):
                    self.videos.append((os.path.join(cls_dir, vid_file), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.videos)

    def extract_frames(self, video_path):
         
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        # Calculate the exact frame numbers we want to extract
        step = max(1, total_frames // self.num_frames)
        target_indices = {i * step for i in range(self.num_frames)}
        
        current_frame = 0
        while True:
            ret, frame = cap.read()
            
            # Stop if the video ends or we've passed our highest target frame
            if not ret or current_frame > max(target_indices, default=0):
                break
                
            # If the current frame is one of our targets, save it
            if current_frame in target_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
            current_frame += 1
            
        cap.release()
        
        # Pad with black frames if the video was corrupted or too short
        while len(frames) < self.num_frames:
            #Used np.uint8 instead of default float32 to prevent TorchVision transform crashes
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
        return frames[:self.num_frames]
    
    def __getitem__(self, idx):
        vid_path, label = self.videos[idx]
        frames = self.extract_frames(vid_path)
        
        if self.transform:
            # Transform each frame and stack into (Frames, Channels, Height, Width)
            frames = torch.stack([self.transform(T.ToTensor()(f)) for f in frames])
            
        return frames, label

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
class VideoResNet(nn.Module):
    """Extracts features using ResNet-18 with Dropout and Frozen Layers to prevent overfitting."""
    def __init__(self, embedding_dim=512):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)
        
        #Freeze the early layers (Layer 1 and Layer 2)
        # This preserves basic edge/texture detection and reduces trainable parameters
        for param in resnet.conv1.parameters(): param.requires_grad = False
        for param in resnet.bn1.parameters(): param.requires_grad = False
        for param in resnet.layer1.parameters(): param.requires_grad = False
        for param in resnet.layer2.parameters(): param.requires_grad = False
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        #Add Dropout before projection
        # ResNet-18 outputs 512 dimensions
        self.dropout = nn.Dropout(p=0.5)
        self.projector = nn.Linear(512, embedding_dim)
        
    def forward(self, x):
        # x shape: (Batch, Frames, Channels, Height, Width)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        # Extract features per frame
        features = self.backbone(x) # (B*T, 512, 1, 1)
        features = features.view(B, T, -1) # (B, T, 512)
        
        # Average across the temporal dimension (frames)
        video_features = features.mean(dim=1) 
        
        # Apply Dropout and Project
        video_features = self.dropout(video_features)
        embeddings = self.projector(video_features)
        
        return embeddings
    
class VideoResNet3D(nn.Module):
    """Extracts spatial-temporal features using a 3D ResNet-18."""
    def __init__(self, embedding_dim=512):
        super().__init__()
        # Load pre-trained 3D ResNet-18 
        weights = models.video.R3D_18_Weights.DEFAULT
        resnet3d = models.video.r3d_18(weights=weights)
        
        # Freeze early layers to prevent overfitting
        for param in resnet3d.stem.parameters(): param.requires_grad = False
        for param in resnet3d.layer1.parameters(): param.requires_grad = False
        for param in resnet3d.layer2.parameters(): param.requires_grad = False
        
        # Remove the final fully connected classification layer
        self.backbone = nn.Sequential(*list(resnet3d.children())[:-1])
        
        # Add Dropout before projection
        self.dropout = nn.Dropout(p=0.5)
        self.projector = nn.Linear(512, embedding_dim)
        
    def forward(self, x):

        #Dataset outputs shape: (Batch, Frames, Channels, Height, Width)
        # 3D ResNet STRICTLY expects: (Batch, Channels, Frames, Height, Width)
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        
        # Extract 3D features (Output shape: B, 512, 1, 1, 1)
        features = self.backbone(x) 
        
        # Flatten the temporal/spatial dimensions down to just (B, 512)
        video_features = features.view(B, -1) 
        
        # Apply Dropout and Project to ArcFace embedding space
        video_features = self.dropout(video_features)
        embeddings = self.projector(video_features)
        
        return embeddings
    
class ArcFaceLayer(nn.Module):
    """ArcFace Margin layer for discriminative training."""
    def __init__(self, in_features, num_classes, s=20.0, m=0.30):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize features and weights
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        
        # Add margin: cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * math.cos(self.m) - sine * math.sin(self.m)
        
        # Create one-hot mask
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin only to ground-truth classes
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
    
# ==========================================
# TRAINING & EXTRACTION
# ==========================================
def train_arcface(model, arcface_layer, dataloader, epochs, device, save_prefix="2d_model"):
    """Trains the embedding model using ArcFace loss."""
    model.train()
    arcface_layer.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(arcface_layer.parameters()), lr=1e-4
    )

    logging.info(f"Starting ArcFace Training for {save_prefix}")
    for epoch in range(epochs):
        total_loss = 0
        for frames, labels in dataloader:
            frames, labels = frames.to(device), labels.to(device)
            
            optimizer.zero_grad()
            embeddings = model(frames)
            outputs = arcface_layer(embeddings, labels)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

    model_dir = os.environ.get("TORCH_HOME", "./")
    os.makedirs(model_dir, exist_ok=True)
    logging.info(f"Saving trained models to {model_dir}")

  
    model_path = os.path.join(model_dir, f"{save_prefix}_resnet.pth")
    torch.save(model.state_dict(), model_path)

    arcface_path = os.path.join(model_dir, f"{save_prefix}_arcface.pth")
    torch.save(arcface_layer.state_dict(), arcface_path)

@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """Passes videos through the trained model to get final embeddings."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    logging.info("Extracting final embeddings")
    for frames, labels in dataloader:
        frames = frames.to(device)
        embeddings = model(frames)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())
        
    return torch.cat(all_embeddings).numpy(), torch.cat(all_labels).numpy()

def evaluate_on_holdout_set(train_embeddings, train_labels, test_embeddings, test_labels, class_names, n_neighbors=7):
    """Trains a k-NN on the training set and evaluates its performance on a holdout test set."""
    logging.info(f"Evaluating on Holdout Test Set with k-NN (k={n_neighbors})...")
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(train_embeddings, train_labels)
    y_pred = knn.predict(test_embeddings)
    
    report_str = classification_report(
        test_labels,
        y_pred, 
        labels=range(len(class_names)), 
        target_names=class_names, 
        zero_division=0
    )
    report_dict = classification_report(
        test_labels, 
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names, 
        output_dict=True, 
        zero_division=0
    )
    
    logging.info("Holdout Set Evaluation Complete.")
    print(report_str)
    
    return pd.DataFrame(report_dict).T, knn

def evaluate_on_transformed_data(knn_classifier, model, test_videos, transformed_data_dir, dataset, device, transform):
    """
    Evaluates a trained model and k-NN classifier on a directory of transformed videos.

    This function iterates through the transformed data, finds files corresponding to the
    holdout set, runs predictions, and compiles the results.
    """
    results = []
    all_embeddings = []
    all_labels = []
    model.eval()

    # Create a map from original filename (without extension) to its true label index
    holdout_map = {os.path.splitext(os.path.basename(path))[0]: label for path, label in test_videos}

    logging.info("Scanning transformed data directory for holdout set videos...")
    for root, _, files in os.walk(transformed_data_dir):
        for file in files:
            file_basename_no_ext = os.path.splitext(file)[0]

            if file_basename_no_ext in holdout_map:
                transformed_path = os.path.join(root, file)
                actual_label_idx = holdout_map[file_basename_no_ext]
                pipeline_name = os.path.basename(root)

                # Extract frames, create tensor, and move to device
                frames = dataset.extract_frames(transformed_path)
                if not frames or len(frames) < dataset.num_frames:
                    logging.warning(f"Could not extract sufficient frames from {transformed_path}, skipping.")
                    continue

                frames_tensor = torch.stack([transform(T.ToTensor()(f)) for f in frames])
                frames_tensor = frames_tensor.unsqueeze(0).to(device)

                # Adjust tensor shape for 3D models
                if isinstance(model, VideoResNet3D):
                    frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4)

                # Get embedding and predict with k-NN
                with torch.no_grad():
                    embedding = model(frames_tensor)
                    embedding = F.normalize(embedding, p=2, dim=1)
                    pred_label_idx = knn_classifier.predict(embedding.cpu().numpy())[0]

                all_embeddings.append(embedding.cpu().numpy())
                all_labels.append(actual_label_idx)

                results.append({
                    "pipeline": pipeline_name,
                    "predicted_label": dataset.classes[pred_label_idx],
                    "actual_label": dataset.classes[actual_label_idx],
                    "predicted_label_idx": pred_label_idx,
                    "actual_label_idx": actual_label_idx,
                })

    return pd.DataFrame(results), np.concatenate(all_embeddings), np.array(all_labels)

def generate_report_from_results(results_df, class_names):
    """Generates a classification report DataFrame from a raw results DataFrame."""
    report_dict = classification_report(
        results_df['actual_label_idx'],
        results_df['predicted_label_idx'],
        labels=range(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    return pd.DataFrame(report_dict).T

def save_evaluation_results(report_df, output_path, header="Holdout Test Set Results"):
    """Saves holdout evaluation DataFrames to a single CSV file."""
    try:
        with open(output_path, 'w') as f:
            if header:
                f.write(f"{header}\n")
            report_df.to_csv(f)
        logging.info(f"Evaluation results saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save evaluation results: {e}")

def calculate_cluster_distances(embeddings, labels, class_names, output_path="cluster_distances.csv"):
    """
    Calculates the pairwise distance between the normalized centroids of each cluster
    in the ArcFace hypersphere and saves a matrix to CSV.
    """
    logging.info("Calculating ArcFace cluster centroid distances")
    
    unique_labels = np.unique(labels)
    centroids = []
    valid_class_names = []
    
    #Find the true center of each class on the hypersphere
    for label in unique_labels:
        # Get all embeddings belonging to this specific generator
        class_embeddings = embeddings[labels == label]
        
        #Calculate the raw mathematical mean
        mean_vector = np.mean(class_embeddings, axis=0)
        
        #L2 Normalise to push the centroid back onto the hypersphere surface
        normalised_centroid = mean_vector / np.linalg.norm(mean_vector)
        
        centroids.append(normalised_centroid)
        
        #map the label index to the string name 
        raw_name = class_names[label]
        valid_class_names.append(LABEL_MAP.get(raw_name, raw_name)) 
        
    centroids = np.array(centroids)
    
    # Calculate pairwise distances - We use Cosine Distance (1 - cosine similarity) because ArcFace optimizes angles!
    dist_matrix = cdist(centroids, centroids, metric='cosine')
    

    dist_df = pd.DataFrame(dist_matrix, index=valid_class_names, columns=valid_class_names)
    
    try:
        dist_df.to_csv(output_path)
        logging.info(f"Cluster distance matrix saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save cluster distances: {e}")
        
    return dist_df


# ==========================================
# VISUALISATION 
# ==========================================
def plot_clusters(embeddings, labels, class_names, video_paths=None, output_path="clusters.png"):
    """Uses t-SNE to reduce embeddings to 2D and saves a publication-ready static plot."""
    logging.info("Running t-SNE reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Map the numeric labels back to their actual string names
    label_names = [class_names[lbl] for lbl in labels]

    # Create a DataFrame for Seaborn
    df = pd.DataFrame({
        't-SNE Component 1': embeddings_2d[:, 0],
        't-SNE Component 2': embeddings_2d[:, 1],
        'Generator Model': label_names
    })
    df['Generator Model'] = df['Generator Model'].map(LABEL_MAP)  # Map to nicer names for the legend

 
    # Set a clean, academic style
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    
    # Create the canvas (10x8 inches is a good standard size)
    plt.figure(figsize=(10, 8))

    # Generate the scatter plot
    scatter = sns.scatterplot(
        data=df, 
        x='t-SNE Component 1', 
        y='t-SNE Component 2', 
        hue='Generator Model',
        palette='deep',     # A professional, colorblind-friendly palette
        s=80,               # Make the dots slightly larger
        alpha=0.8,          # 80% opacity so overlapping dots create density
        edgecolor='white',  # Crisp white borders around each dot
        linewidth=0.5
    )

    plt.title("ArcFace Video Embeddings (t-SNE Projection)", pad=20, fontsize=14,)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    plt.legend(
        title="Generator Model", 
        title_fontsize='13', 
        fontsize='11', 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', 
        frameon=False
    )

 
    sns.despine()

    plt.tight_layout()
    
    #300 DPI (Dots Per Inch) is the standard for print/PDFs
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() # Free up your system memory
    
    logging.info(f"Publication-ready cluster plot saved to {output_path}")
