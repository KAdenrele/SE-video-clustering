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
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import logging


class VideoDirectoryDataset(Dataset):
    """Loads videos from a directory where subfolders represent class labels."""
    def __init__(self, root_dir, num_frames=10, transform=None):
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
        
        # Step 3: Add Dropout before projection
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
def train_arcface(model, arcface_layer, dataloader, epochs, device):
    """Trains the embedding model using ArcFace loss."""
    model.train()
    arcface_layer.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(arcface_layer.parameters()), lr=1e-4
    )

    logging.info("Starting ArcFace Training")
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

    model_dir = os.environ.get("TORCH_HOME")
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        logging.info(f"Saving trained models to {model_dir}")

        model_path = os.path.join(model_dir, "finetuned_video_resnet.pth")
        torch.save(model.state_dict(), model_path)

        arcface_path = os.path.join(model_dir, "finetuned_arcface_layer.pth")
        torch.save(arcface_layer.state_dict(), arcface_path)
    else:
        logging.warning("TORCH_HOME environment variable not set. Trained models will not be saved.")

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

def evaluate_with_knn_kfold(embeddings, labels, class_names, n_splits=5, n_neighbors=7):
    """
    Evaluates embedding quality using Stratified K-Fold CV with k-NN, 
    ensuring all classes are proportionally represented in every fold.
    """
    logging.info(f"Starting {n_splits}-Fold Stratified Cross-Validation with k-NN (k={n_neighbors})...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    
    fold_results = []
    
    for fold, (train_index, test_index) in enumerate(skf.split(embeddings, labels)):
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        report = classification_report(
            y_test, 
            y_pred, 
            labels=range(len(class_names)), 
            target_names=class_names, 
            output_dict=True, 
            zero_division=0
        )
        
        logging.info(f"  Fold {fold+1}/{n_splits} | Accuracy: {report['accuracy']:.4f}")
        fold_results.append(report)
        
    # Average the metrics across all folds
    avg_report = {}
    for label in class_names:
        avg_report[label] = {
            'precision': np.mean([fr[label]['precision'] for fr in fold_results if label in fr]),
            'recall': np.mean([fr[label]['recall'] for fr in fold_results if label in fr]),
            'f1-score': np.mean([fr[label]['f1-score'] for fr in fold_results if label in fr]),
            'support': np.mean([fr[label]['support'] for fr in fold_results if label in fr])
        }
    
    avg_report['accuracy'] = np.mean([fr['accuracy'] for fr in fold_results])
    for avg_type in ['macro avg', 'weighted avg']:
        avg_report[avg_type] = {
            'precision': np.mean([fr[avg_type]['precision'] for fr in fold_results]),
            'recall': np.mean([fr[avg_type]['recall'] for fr in fold_results]),
            'f1-score': np.mean([fr[avg_type]['f1-score'] for fr in fold_results]),
        }
    
    report_df = pd.DataFrame(avg_report).T
    
    logging.info("Stratified K-Fold Cross-Validation Complete. Average Metrics:")
    print("\n--- Stratified K-Fold Cross-Validation Mean Results ---")
    print(report_df.to_string())
    print("-" * 50)
    
    return report_df

def evaluate_on_holdout_set(train_embeddings, train_labels, test_embeddings, test_labels, class_names, n_neighbors=7):
    """Trains a k-NN on the training set and evaluates its performance on a holdout test set."""
    logging.info(f"Evaluating on Holdout Test Set with k-NN (k={n_neighbors})...")
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(train_embeddings, train_labels)
    y_pred = knn.predict(test_embeddings)
    
    report_str = classification_report(
        test_labels, 
        y_pred, 
        labels=range(len(class_names)), # <-- Added this
        target_names=class_names, 
        zero_division=0
    )
    report_dict = classification_report(
        test_labels, 
        y_pred, 
        labels=range(len(class_names)), # <-- Added this
        target_names=class_names, 
        output_dict=True, 
        zero_division=0
    )
    
    logging.info("Holdout Set Evaluation Complete.")
    print("\n--- Holdout Test Set Results ---")
    print(report_str)
    print("-" * 50)
    
    return pd.DataFrame(report_dict).T

def save_evaluation_results(kfold_df, holdout_df, output_path):
    """Saves the K-Fold and Holdout evaluation DataFrames to a single CSV file."""
    try:
        with open(output_path, 'w') as f:
            f.write("K-Fold Cross-Validation Mean Results\n")
            kfold_df.to_csv(f)
            f.write("\n\nHoldout Test Set Results\n")
            holdout_df.to_csv(f)
        logging.info(f"Evaluation results saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save evaluation results: {e}")

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

    plt.title("ArcFace Video Embeddings (t-SNE Projection)", pad=20, fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Component 1", fontweight='bold')
    plt.ylabel("t-SNE Component 2", fontweight='bold')

    plt.legend(
        title="Generator Model", 
        title_fontsize='13', 
        fontsize='11', 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', 
        frameon=False
    )

    # Remove the top and right box borders for a modern, clean look
    sns.despine()

    plt.tight_layout()
    
    #300 DPI (Dots Per Inch) is the standard for print/PDFs
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() # Free up your system memory
    
    logging.info(f"Publication-ready cluster plot saved to {output_path}")
