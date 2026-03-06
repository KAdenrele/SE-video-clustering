import os
import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ==========================================
# 1. DATASET & VIDEO PROCESSING
# ==========================================
class VideoDirectoryDataset(Dataset):
    """Loads videos from a directory where subfolders represent class labels."""
    def __init__(self, root_dir, num_frames=16, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
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
        
        # Sample frames evenly
        step = max(1, total_frames // self.num_frames)
        for i in range(self.num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # Pad with zeros if video is too short
                frames.append(torch.zeros((224, 224, 3)).numpy())
        cap.release()
        return frames

    def __getitem__(self, idx):
        vid_path, label = self.videos[idx]
        frames = self.extract_frames(vid_path)
        
        if self.transform:
            # Transform each frame and stack into (Frames, Channels, Height, Width)
            frames = torch.stack([self.transform(T.ToTensor()(f)) for f in frames])
            
        return frames, label

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class VideoResNet80(nn.Module):
    """Extracts features using ResNet-80 and averages them across frames."""
    def __init__(self, embedding_dim=512):
        super().__init__()
        weights = models.ResNet80_Weights.DEFAULT
        resnet = models.resnet80(weights=weights)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Project 2048-dim ResNet features to desired embedding space
        self.projector = nn.Linear(2048, embedding_dim)
        
    def forward(self, x):
        # x shape: (Batch, Frames, Channels, Height, Width)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        # Extract features per frame
        features = self.backbone(x) # (B*T, 2048, 1, 1)
        features = features.view(B, T, -1) # (B, T, 2048)
        
        # Average across the temporal dimension (frames)
        video_features = features.mean(dim=1) 
        embeddings = self.projector(video_features)
        
        return embeddings

class ArcFaceLayer(nn.Module):
    """ArcFace Margin layer for discriminative training."""
    def __init__(self, in_features, num_classes, s=30.0, m=0.50):
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
# 3. TRAINING & EXTRACTION
# ==========================================
def train_arcface(model, arcface_layer, dataloader, epochs, device):
    """Trains the embedding model using ArcFace loss."""
    model.train()
    arcface_layer.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(arcface_layer.parameters()), lr=1e-4
    )

    print("Starting ArcFace Training...")
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
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """Passes videos through the trained model to get final embeddings."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    print("Extracting final embeddings...")
    for frames, labels in dataloader:
        frames = frames.to(device)
        embeddings = model(frames)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())
        
    return torch.cat(all_embeddings).numpy(), torch.cat(all_labels).numpy()

# ==========================================
# 4. VISUALIZATION
# ==========================================
def plot_clusters(embeddings, labels, class_names, output_path="clusters.png"):
    """Uses t-SNE to reduce embeddings to 2D and saves a plot."""
    print("Running t-SNE reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10) 
               for i in range(len(class_names))]
    plt.legend(handles, class_names, title="Classes")
    
    plt.title("ArcFace Video Embeddings (t-SNE Projection)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(output_path)
    print(f"Cluster visualization saved to {output_path}")

# ==========================================
# 5. ORCHESTRATION (MAIN)
# ==========================================
if __name__ == "__main__":
    # Settings
    DATA_DIR = "./video_data" # Change this to your directory path
    EPOCHS = 10
    BATCH_SIZE = 8
    EMBEDDING_DIM = 512
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard ImageNet transforms
    transform = T.Compose([
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Setup Data
    # NOTE: Ensure your directories look like: ./video_data/class1/vid.mp4, ./video_data/class2/vid.mp4
    dataset = VideoDirectoryDataset(DATA_DIR, num_frames=16, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    num_classes = len(dataset.classes)

    # Initialize Model & ArcFace
    model = VideoResNet80(embedding_dim=EMBEDDING_DIM).to(device)
    arcface_layer = ArcFaceLayer(in_features=EMBEDDING_DIM, num_classes=num_classes).to(device)

    # 1. Train the model to cluster well
    train_arcface(model, arcface_layer, dataloader, epochs=EPOCHS, device=device)

    # 2. Extract the features
    dataloader_eval = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    embeddings, labels = extract_embeddings(model, dataloader_eval, device)

    # 3. Visualize
    plot_clusters(embeddings, labels, dataset.classes, output_path="video_cluster/cluster_output.png")