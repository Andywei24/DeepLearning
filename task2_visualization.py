import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class FashionNetT2(nn.Module):
    def __init__(self, embedding_dim=128):
        super(FashionNetT2, self).__init__()
        # Convolutional layers remain the same
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Calculate the flattened size based on your new image dimensions
        # You'll need to replace this with the correct size based on your input
        self.fc1 = nn.Linear(8960, 512)  # Modified for your new image size
        self.fc2 = nn.Linear(512, embedding_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def extract_features(model, dataloader, device):
    """Extract features using the model"""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Extracting features"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu())
            labels.extend(targets.cpu().numpy())
    
    features = torch.cat(features).numpy()
    return features, np.array(labels)

def apply_tsne_reduction(features, labels, label_id_to_label_name={}, title_suffix=""):
    """
    Apply t-SNE (t-Distributed Stochastic Neighbor Embedding) dimensionality reduction.
    t-SNE is a non-linear dimensionality reduction technique that is particularly good at 
    preserving local structure in the data, making it effective for visualization of high-dimensional data.
    """
    
    print(f"Applying t-SNE to {len(features)} samples...")
    
    # Adjust perplexity based on dataset size
    perplexity = min(30, len(features) // 4)
    perplexity = max(5, perplexity)  # Ensure minimum perplexity
    reducer = TSNE(
        n_components=2, 
        random_state=42, 
        perplexity=perplexity, 
        n_iter=1000,
        n_jobs=1  # Use single thread to avoid CPU overload
    )
    reduced_features = reducer.fit_transform(features)
        
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    # Get unique classes and create color map
    unique_classes = sorted(set(labels))
    colors = plt.cm.tab10(range(len(unique_classes)))
    
    for i, class_id in enumerate(unique_classes):
        mask = np.array(labels) == class_id
        if np.any(mask):
            class_name = label_id_to_label_name.get(class_id, f"Class {class_id}")
            plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1], 
                       c=[colors[i]], label=f"{class_name} ({class_id})", alpha=0.7, s=30)
    
    plt.title(f't-SNE Visualization of Task 2 Features - {title_suffix}', fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return reduced_features

# Example usage:
if __name__ == "__main__":
    # Initialize model with the new architecture
    model = FashionNetT2(embedding_dim=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Since we can't load the old weights directly, we'll need to train the model again
    # or use a different approach to transfer the weights
    
    # Example of how to use the visualization:
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("=== TRAIN DATASET VISUALIZATION ===")
    train_features, train_labels = extract_features(model, train_loader, device)
    train_tsne = apply_tsne_reduction(train_features, train_labels, title_suffix='Train Dataset')
    
    print("\n=== TEST DATASET VISUALIZATION ===")
    test_features, test_labels = extract_features(model, test_loader, device)
    test_tsne = apply_tsne_reduction(test_features, test_labels, title_suffix='Test Dataset') 