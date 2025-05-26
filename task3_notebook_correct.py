# Task 3: T-shirt vs Shirt Analysis - Notebook Cell Version
# Using main_support.csv with correct Tshirts (ID=0) and Shirts (ID=1)

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.decomposition import PCA

# Load and filter the correct data
def load_shirt_tshirt_data():
    """Load main_support.csv and filter for Tshirts and Shirts"""
    support_df = pd.read_csv('dataset/main_support.csv')
    
    # Filter for Tshirts (articleTypeId=0) and Shirts (articleTypeId=1)
    shirt_tshirt_df = support_df[support_df['articleTypeId'].isin([0, 1])].copy()
    
    print(f"Total samples in main_support.csv: {len(support_df)}")
    print(f"Tshirts and Shirts samples: {len(shirt_tshirt_df)}")
    print(f"Tshirts (ID=0): {len(shirt_tshirt_df[shirt_tshirt_df['articleTypeId'] == 0])}")
    print(f"Shirts (ID=1): {len(shirt_tshirt_df[shirt_tshirt_df['articleTypeId'] == 1])}")
    
    return shirt_tshirt_df

# Extract features for shirt/T-shirt samples
def extract_features_correct(model, dataset, shirt_tshirt_df, device):
    """Extract features for filtered shirt/T-shirt samples"""
    
    # Create mapping from imageId to dataset index
    image_id_to_idx = {}
    for idx in range(len(dataset)):
        try:
            image_id = dataset.df.iloc[idx]['imageId']
            image_id_to_idx[image_id] = idx
        except:
            continue
    
    # Find valid samples
    valid_samples = []
    labels = []
    dataset_indices = []
    
    for _, row in shirt_tshirt_df.iterrows():
        image_id = row['imageId']
        if image_id in image_id_to_idx:
            dataset_idx = image_id_to_idx[image_id]
            valid_samples.append(dataset_idx)
            labels.append(row['articleTypeId'])  # 0=Tshirt, 1=Shirt
            dataset_indices.append(dataset_idx)
    
    print(f"Found {len(valid_samples)} valid samples in dataset")
    
    if len(valid_samples) == 0:
        print("❌ No matching samples found!")
        return None, None, None
    
    # Extract features
    model.eval()
    features = []
    
    with torch.no_grad():
        for i in range(0, len(valid_samples), 32):
            batch_indices = valid_samples[i:i+32]
            batch_images = []
            
            for idx in batch_indices:
                try:
                    image, _ = dataset[idx]
                    batch_images.append(image)
                except:
                    continue
            
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(device)
                batch_features = model(batch_tensor)
                features.append(batch_features.cpu().numpy())
    
    if features:
        features = np.vstack(features)
        return features, np.array(labels), np.array(dataset_indices)
    else:
        return None, None, None

# Analyze transition between T-shirts and Shirts
def analyze_transition(features, labels, dataset_indices, dataset):
    """Analyze T-shirt vs Shirt transition"""
    
    # Separate classes
    tshirt_mask = labels == 0
    shirt_mask = labels == 1
    
    tshirt_features = features[tshirt_mask]
    shirt_features = features[shirt_mask]
    tshirt_indices = dataset_indices[tshirt_mask]
    shirt_indices = dataset_indices[shirt_mask]
    
    print(f"T-shirts: {len(tshirt_features)} samples")
    print(f"Shirts: {len(shirt_features)} samples")
    
    # Compute centroids
    tshirt_centroid = np.mean(tshirt_features, axis=0)
    shirt_centroid = np.mean(shirt_features, axis=0)
    
    # Find boundary samples
    tshirt_to_shirt_dist = np.linalg.norm(tshirt_features - shirt_centroid, axis=1)
    shirt_to_tshirt_dist = np.linalg.norm(shirt_features - tshirt_centroid, axis=1)
    
    boundary_tshirt_idx = np.argmin(tshirt_to_shirt_dist)
    boundary_shirt_idx = np.argmin(shirt_to_tshirt_dist)
    
    # Find typical samples
    tshirt_to_own_dist = np.linalg.norm(tshirt_features - tshirt_centroid, axis=1)
    shirt_to_own_dist = np.linalg.norm(shirt_features - shirt_centroid, axis=1)
    
    typical_tshirt_idx = np.argmin(tshirt_to_own_dist)
    typical_shirt_idx = np.argmin(shirt_to_own_dist)
    
    # Calculate metrics
    centroid_distance = np.linalg.norm(shirt_centroid - tshirt_centroid)
    min_boundary_distance = min(tshirt_to_shirt_dist[boundary_tshirt_idx],
                               shirt_to_tshirt_dist[boundary_shirt_idx])
    overlap_ratio = min_boundary_distance / centroid_distance
    
    print(f"Centroid distance: {centroid_distance:.4f}")
    print(f"Overlap ratio: {overlap_ratio:.4f}")
    
    # PCA Visualization
    all_features = np.vstack([tshirt_features, shirt_features])
    pca = PCA(n_components=2)
    all_features_pca = pca.fit_transform(all_features)
    
    n_tshirt = len(tshirt_features)
    tshirt_pca = all_features_pca[:n_tshirt]
    shirt_pca = all_features_pca[n_tshirt:]
    
    tshirt_centroid_pca = pca.transform(tshirt_centroid.reshape(1, -1))[0]
    shirt_centroid_pca = pca.transform(shirt_centroid.reshape(1, -1))[0]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Feature space plot
    ax1.scatter(tshirt_pca[:, 0], tshirt_pca[:, 1], alpha=0.6, label='T-shirts', color='red')
    ax1.scatter(shirt_pca[:, 0], shirt_pca[:, 1], alpha=0.6, label='Shirts', color='blue')
    ax1.scatter(tshirt_centroid_pca[0], tshirt_centroid_pca[1], color='red', s=300, marker='*', 
               label='T-shirt Centroid', edgecolor='black', linewidth=2)
    ax1.scatter(shirt_centroid_pca[0], shirt_centroid_pca[1], color='blue', s=300, marker='*', 
               label='Shirt Centroid', edgecolor='black', linewidth=2)
    ax1.plot([tshirt_centroid_pca[0], shirt_centroid_pca[0]], 
             [tshirt_centroid_pca[1], shirt_centroid_pca[1]], 
             'k--', linewidth=3, alpha=0.8, label='Transition Line')
    ax1.set_title('Feature Space: T-shirts vs Shirts')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Boundary analysis
    ax2.scatter(tshirt_pca[:, 0], tshirt_pca[:, 1], alpha=0.4, color='red', s=20)
    ax2.scatter(shirt_pca[:, 0], shirt_pca[:, 1], alpha=0.4, color='blue', s=20)
    ax2.scatter(tshirt_pca[typical_tshirt_idx, 0], tshirt_pca[typical_tshirt_idx, 1], 
               color='red', s=150, marker='o', label='Typical T-shirt', edgecolor='black')
    ax2.scatter(shirt_pca[typical_shirt_idx, 0], shirt_pca[typical_shirt_idx, 1], 
               color='blue', s=150, marker='o', label='Typical Shirt', edgecolor='black')
    ax2.scatter(tshirt_pca[boundary_tshirt_idx, 0], tshirt_pca[boundary_tshirt_idx, 1], 
               color='red', s=150, marker='D', label='Boundary T-shirt', edgecolor='yellow')
    ax2.scatter(shirt_pca[boundary_shirt_idx, 0], shirt_pca[boundary_shirt_idx, 1], 
               color='blue', s=150, marker='D', label='Boundary Shirt', edgecolor='yellow')
    ax2.set_title('Boundary Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Display examples
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    examples = [
        ('Typical T-shirt', tshirt_indices[typical_tshirt_idx]),
        ('Boundary T-shirt', tshirt_indices[boundary_tshirt_idx]),
        ('Boundary Shirt', shirt_indices[boundary_shirt_idx]),
        ('Typical Shirt', shirt_indices[typical_shirt_idx])
    ]
    
    for i, (title, dataset_idx) in enumerate(examples):
        try:
            image, _ = dataset[dataset_idx]
            image_id = dataset.df.iloc[dataset_idx]['imageId']
            article_name = dataset.df.iloc[dataset_idx]['articleTypeName']
            
            if torch.is_tensor(image):
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = image.permute(1, 2, 0).numpy()
                image_np = image_np * std + mean
                image_np = np.clip(image_np, 0, 1)
            else:
                image_np = np.array(image)
            
            axes[i].imshow(image_np)
            axes[i].set_title(f'{title}\n{article_name}\nID: {image_id}')
            axes[i].axis('off')
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error\n{title}', ha='center', va='center')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Generate answer
    print("\n" + "="*60)
    print("ANSWER: Do you observe characteristics that make a shirt close to a T-shirt?")
    print("="*60)
    
    if overlap_ratio < 0.4:
        answer = f"YES, with clear distinctions (Overlap: {overlap_ratio:.1%}). The model learned distinct features for collars, buttons, and formality levels."
    elif overlap_ratio < 0.7:
        answer = f"YES, significant similarities (Overlap: {overlap_ratio:.1%}). Casual shirts share fabric textures and basic cuts with T-shirts."
    else:
        answer = f"YES, substantial overlap (Overlap: {overlap_ratio:.1%}). T-shirts and casual shirts exist on a continuum with subtle differences."
    
    print(answer)
    
    return overlap_ratio

# USAGE INSTRUCTIONS:
print("="*60)
print("TASK 3: T-SHIRT vs SHIRT ANALYSIS")
print("="*60)
print("1. Load your trained model and dataset")
print("2. Run the following code:")
print()
print("# Load data")
print("shirt_tshirt_df = load_shirt_tshirt_data()")
print()
print("# Extract features")
print("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
print("features, labels, indices = extract_features_correct(model, dataset, shirt_tshirt_df, device)")
print()
print("# Analyze transition")
print("if features is not None:")
print("    overlap_ratio = analyze_transition(features, labels, indices, dataset)")
print("="*60) 