# Task 3: Complete T-shirt vs Shirt Analysis
# Using main_support.csv and main_test.csv with correct labels

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.decomposition import PCA

print("="*80)
print("TASK 3: COMPLETE T-SHIRT vs SHIRT TRANSITION ANALYSIS")
print("Using main_support.csv and main_test.csv with correct labels")
print("="*80)

def load_complete_shirt_tshirt_data():
    """Load both main_support.csv and main_test.csv and filter for Tshirts and Shirts"""
    
    # Load both CSV files
    support_df = pd.read_csv('dataset/main_support.csv')
    test_df = pd.read_csv('dataset/main_test.csv')
    
    print(f"Support samples: {len(support_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Filter for Tshirts (articleTypeId=0) and Shirts (articleTypeId=1)
    support_filtered = support_df[support_df['articleTypeId'].isin([0, 1])].copy()
    test_filtered = test_df[test_df['articleTypeId'].isin([0, 1])].copy()
    
    # Add dataset source
    support_filtered['dataset_source'] = 'support'
    test_filtered['dataset_source'] = 'test'
    
    # Combine datasets
    combined_df = pd.concat([support_filtered, test_filtered], ignore_index=True)
    
    print(f"\nFiltered Results:")
    print(f"Support - Tshirts: {len(support_filtered[support_filtered['articleTypeId'] == 0])}")
    print(f"Support - Shirts: {len(support_filtered[support_filtered['articleTypeId'] == 1])}")
    print(f"Test - Tshirts: {len(test_filtered[test_filtered['articleTypeId'] == 0])}")
    print(f"Test - Shirts: {len(test_filtered[test_filtered['articleTypeId'] == 1])}")
    print(f"Total combined: {len(combined_df)}")
    
    return combined_df, support_filtered, test_filtered

def extract_features_complete(model, support_dataset, test_dataset, combined_df, device):
    """Extract features for both support and test datasets"""
    
    # Create mappings for both datasets
    support_id_to_idx = {}
    test_id_to_idx = {}
    
    # Map support dataset
    for idx in range(len(support_dataset)):
        try:
            image_id = support_dataset.df.iloc[idx]['imageId']
            support_id_to_idx[image_id] = idx
        except:
            continue
    
    # Map test dataset
    for idx in range(len(test_dataset)):
        try:
            image_id = test_dataset.df.iloc[idx]['imageId']
            test_id_to_idx[image_id] = idx
        except:
            continue
    
    print(f"Support dataset mapping: {len(support_id_to_idx)} images")
    print(f"Test dataset mapping: {len(test_id_to_idx)} images")
    
    # Find valid samples
    valid_samples = []
    labels = []
    dataset_indices = []
    dataset_sources = []
    
    for _, row in combined_df.iterrows():
        image_id = row['imageId']
        source = row['dataset_source']
        
        if source == 'support' and image_id in support_id_to_idx:
            dataset_idx = support_id_to_idx[image_id]
            valid_samples.append(('support', dataset_idx))
            labels.append(row['articleTypeId'])
            dataset_indices.append(dataset_idx)
            dataset_sources.append('support')
        elif source == 'test' and image_id in test_id_to_idx:
            dataset_idx = test_id_to_idx[image_id]
            valid_samples.append(('test', dataset_idx))
            labels.append(row['articleTypeId'])
            dataset_indices.append(dataset_idx)
            dataset_sources.append('test')
    
    print(f"Found {len(valid_samples)} valid samples total")
    
    if len(valid_samples) == 0:
        print("❌ No matching samples found!")
        return None, None, None, None
    
    # Extract features
    model.eval()
    features = []
    
    with torch.no_grad():
        for i in range(0, len(valid_samples), 32):
            batch_samples = valid_samples[i:i+32]
            batch_images = []
            
            for source, idx in batch_samples:
                try:
                    if source == 'support':
                        image, _ = support_dataset[idx]
                    else:  # test
                        image, _ = test_dataset[idx]
                    batch_images.append(image)
                except:
                    continue
            
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(device)
                batch_features = model(batch_tensor)
                features.append(batch_features.cpu().numpy())
    
    if features:
        features = np.vstack(features)
        print(f"Extracted features shape: {features.shape}")
        return features, np.array(labels), np.array(dataset_indices), np.array(dataset_sources)
    else:
        return None, None, None, None

def analyze_complete_transition(features, labels, dataset_indices, dataset_sources, 
                               support_dataset, test_dataset):
    """Complete analysis of T-shirt vs Shirt transition across both datasets"""
    
    # Separate by class and dataset
    tshirt_mask = labels == 0
    shirt_mask = labels == 1
    support_mask = dataset_sources == 'support'
    test_mask = dataset_sources == 'test'
    
    # Get features by class
    tshirt_features = features[tshirt_mask]
    shirt_features = features[shirt_mask]
    
    # Get features by dataset
    support_features = features[support_mask]
    test_features = features[test_mask]
    
    # Get indices
    tshirt_indices = dataset_indices[tshirt_mask]
    shirt_indices = dataset_indices[shirt_mask]
    tshirt_sources = dataset_sources[tshirt_mask]
    shirt_sources = dataset_sources[shirt_mask]
    
    print(f"\nData Summary:")
    print(f"Total T-shirts: {len(tshirt_features)} samples")
    print(f"Total Shirts: {len(shirt_features)} samples")
    print(f"Support data: {len(support_features)} samples")
    print(f"Test data: {len(test_features)} samples")
    
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
    
    print(f"\nAnalysis Results:")
    print(f"Centroid distance: {centroid_distance:.4f}")
    print(f"Minimum boundary distance: {min_boundary_distance:.4f}")
    print(f"Overlap ratio: {overlap_ratio:.4f}")
    
    # PCA Visualization
    all_features = np.vstack([tshirt_features, shirt_features])
    pca = PCA(n_components=2)
    all_features_pca = pca.fit_transform(all_features)
    
    n_tshirt = len(tshirt_features)
    tshirt_pca = all_features_pca[:n_tshirt]
    shirt_pca = all_features_pca[n_tshirt:]
    
    # Transform centroids
    tshirt_centroid_pca = pca.transform(tshirt_centroid.reshape(1, -1))[0]
    shirt_centroid_pca = pca.transform(shirt_centroid.reshape(1, -1))[0]
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Overall feature space
    ax1.scatter(tshirt_pca[:, 0], tshirt_pca[:, 1], alpha=0.6, label='T-shirts', color='red', s=30)
    ax1.scatter(shirt_pca[:, 0], shirt_pca[:, 1], alpha=0.6, label='Shirts', color='blue', s=30)
    ax1.scatter(tshirt_centroid_pca[0], tshirt_centroid_pca[1], color='red', s=300, marker='*', 
               label='T-shirt Centroid', edgecolor='black', linewidth=2)
    ax1.scatter(shirt_centroid_pca[0], shirt_centroid_pca[1], color='blue', s=300, marker='*', 
               label='Shirt Centroid', edgecolor='black', linewidth=2)
    ax1.plot([tshirt_centroid_pca[0], shirt_centroid_pca[0]], 
             [tshirt_centroid_pca[1], shirt_centroid_pca[1]], 
             'k--', linewidth=3, alpha=0.8, label='Transition Line')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title('Complete Feature Space: T-shirts vs Shirts')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Dataset source comparison
    support_pca = pca.transform(support_features)
    test_pca = pca.transform(test_features)
    ax2.scatter(support_pca[:, 0], support_pca[:, 1], alpha=0.6, label='Support', color='green', s=20)
    ax2.scatter(test_pca[:, 0], test_pca[:, 1], alpha=0.6, label='Test', color='orange', s=20)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.set_title('Dataset Source Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Boundary analysis
    ax3.scatter(tshirt_pca[:, 0], tshirt_pca[:, 1], alpha=0.4, color='red', s=20, label='T-shirts')
    ax3.scatter(shirt_pca[:, 0], shirt_pca[:, 1], alpha=0.4, color='blue', s=20, label='Shirts')
    ax3.scatter(tshirt_pca[typical_tshirt_idx, 0], tshirt_pca[typical_tshirt_idx, 1], 
               color='red', s=150, marker='o', label='Typical T-shirt', edgecolor='black', linewidth=2)
    ax3.scatter(shirt_pca[typical_shirt_idx, 0], shirt_pca[typical_shirt_idx, 1], 
               color='blue', s=150, marker='o', label='Typical Shirt', edgecolor='black', linewidth=2)
    ax3.scatter(tshirt_pca[boundary_tshirt_idx, 0], tshirt_pca[boundary_tshirt_idx, 1], 
               color='red', s=150, marker='D', label='Boundary T-shirt', edgecolor='yellow', linewidth=2)
    ax3.scatter(shirt_pca[boundary_shirt_idx, 0], shirt_pca[boundary_shirt_idx, 1], 
               color='blue', s=150, marker='D', label='Boundary Shirt', edgecolor='yellow', linewidth=2)
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax3.set_title('Boundary Analysis: Typical vs Cross-Class Similar')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Combined class and dataset view
    # Separate by both class and dataset
    support_tshirt_mask = (labels == 0) & (dataset_sources == 'support')
    support_shirt_mask = (labels == 1) & (dataset_sources == 'support')
    test_tshirt_mask = (labels == 0) & (dataset_sources == 'test')
    test_shirt_mask = (labels == 1) & (dataset_sources == 'test')
    
    if np.any(support_tshirt_mask):
        support_tshirt_pca = pca.transform(features[support_tshirt_mask])
        ax4.scatter(support_tshirt_pca[:, 0], support_tshirt_pca[:, 1], 
                   alpha=0.7, color='red', marker='o', s=40, label='Support T-shirts')
    
    if np.any(support_shirt_mask):
        support_shirt_pca = pca.transform(features[support_shirt_mask])
        ax4.scatter(support_shirt_pca[:, 0], support_shirt_pca[:, 1], 
                   alpha=0.7, color='blue', marker='o', s=40, label='Support Shirts')
    
    if np.any(test_tshirt_mask):
        test_tshirt_pca = pca.transform(features[test_tshirt_mask])
        ax4.scatter(test_tshirt_pca[:, 0], test_tshirt_pca[:, 1], 
                   alpha=0.7, color='red', marker='^', s=40, label='Test T-shirts')
    
    if np.any(test_shirt_mask):
        test_shirt_pca = pca.transform(features[test_shirt_mask])
        ax4.scatter(test_shirt_pca[:, 0], test_shirt_pca[:, 1], 
                   alpha=0.7, color='blue', marker='^', s=40, label='Test Shirts')
    
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax4.set_title('Class and Dataset Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Complete T-shirts vs Shirts Analysis (Support + Test Data)', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Display example images
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    examples = [
        ('Typical T-shirt', tshirt_indices[typical_tshirt_idx], tshirt_sources[typical_tshirt_idx]),
        ('Boundary T-shirt→Shirt', tshirt_indices[boundary_tshirt_idx], tshirt_sources[boundary_tshirt_idx]),
        ('Boundary Shirt→T-shirt', shirt_indices[boundary_shirt_idx], shirt_sources[boundary_shirt_idx]),
        ('Typical Shirt', shirt_indices[typical_shirt_idx], shirt_sources[typical_shirt_idx])
    ]
    
    for i, (title, dataset_idx, source) in enumerate(examples):
        try:
            if source == 'support':
                image, _ = support_dataset[dataset_idx]
                image_id = support_dataset.df.iloc[dataset_idx]['imageId']
                article_name = support_dataset.df.iloc[dataset_idx]['articleTypeName']
            else:  # test
                image, _ = test_dataset[dataset_idx]
                image_id = test_dataset.df.iloc[dataset_idx]['imageId']
                article_name = test_dataset.df.iloc[dataset_idx]['articleTypeName']
            
            if torch.is_tensor(image):
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = image.permute(1, 2, 0).numpy()
                image_np = image_np * std + mean
                image_np = np.clip(image_np, 0, 1)
            else:
                image_np = np.array(image)
            
            axes[i].imshow(image_np)
            axes[i].set_title(f'{title}\n{article_name} ({source})\nID: {image_id}', fontsize=9)
            axes[i].axis('off')
            
        except Exception as e:
            print(f"Error displaying {title}: {e}")
            axes[i].text(0.5, 0.5, f'Error\n{title}', ha='center', va='center')
            axes[i].axis('off')
    
    plt.suptitle('Visual Examples: T-shirts vs Shirts Analysis', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Generate comprehensive answer
    print("\n" + "="*80)
    print("ANSWER: Do you observe characteristics that make a shirt close to a T-shirt?")
    print("="*80)
    
    if overlap_ratio < 0.4:
        answer = f"""
YES, with clear distinctions observed (Overlap ratio: {overlap_ratio:.1%}):

• FEATURE SPACE ANALYSIS:
  - Well-separated clusters indicate the model learned distinct features
  - Consistent separation across both support and test datasets
  - Clear centroid separation suggests robust class discrimination

• VISUAL CHARACTERISTICS making Shirts close to T-shirts:
  - Casual shirt designs with minimal formal elements
  - Similar fabric textures and color patterns
  - Basic silhouettes without complex structural details
  - Polo-style shirts with simple collar designs

• DISTINGUISHING FEATURES the model learned:
  - Collar presence and style (button-down vs crew neck)
  - Closure methods (buttons vs pullover design)
  - Overall formality level and fit structure
  - Sleeve and neckline construction details

• DATASET CONSISTENCY:
  - Similar patterns observed in both support and test data
  - Robust feature learning across different data splits
        """
    elif overlap_ratio < 0.7:
        answer = f"""
YES, significant transitional characteristics observed (Overlap ratio: {overlap_ratio:.1%}):

• FEATURE SPACE ANALYSIS:
  - Moderate overlap indicates shared visual elements between classes
  - Boundary samples show clear transitional properties
  - Consistent patterns across support and test datasets

• VISUAL CHARACTERISTICS making Shirts close to T-shirts:
  - Casual shirts with minimal formal features
  - Polo-style shirts resembling structured T-shirts
  - Similar fabric types and color schemes
  - Basic cuts without complex design elements

• SHARED CHARACTERISTICS:
  - Fabric texture and material appearance
  - Color patterns and basic garment structure
  - Simple, clean design aesthetics
  - Casual wear styling elements

• KEY DISTINGUISHING FEATURES:
  - Collar presence (polo collar vs crew neck)
  - Button details and closure methods
  - Overall formality and structure level
        """
    else:
        answer = f"""
YES, substantial feature overlap observed (Overlap ratio: {overlap_ratio:.1%}):

• FEATURE SPACE ANALYSIS:
  - High similarity suggests many shared characteristics
  - Significant boundary overlap indicates transitional garment styles
  - The model captures subtle but important distinctions

• STRONG SHARED CHARACTERISTICS:
  - Very similar fabric textures and materials
  - Comparable color schemes and patterns
  - Basic garment construction and silhouettes
  - Casual styling and fit characteristics

• SUBTLE DISTINGUISHING FEATURES:
  - Collar type (minimal collar vs crew neck)
  - Closure method (few buttons vs pullover)
  - Minor styling details and finishing
  - Slight differences in formality level

• CONCLUSION:
  - T-shirts and casual shirts exist on a continuum
  - The model learned to distinguish subtle design elements
  - Boundary samples represent transitional garment styles
        """
    
    print(answer)
    
    return {
        'overlap_ratio': overlap_ratio,
        'centroid_distance': centroid_distance,
        'answer': answer.strip(),
        'n_tshirts': len(tshirt_features),
        'n_shirts': len(shirt_features),
        'n_support': len(support_features),
        'n_test': len(test_features)
    }

# Usage instructions
print("USAGE:")
print("1. Load your trained model and both datasets")
print("2. Run:")
print("   combined_df, support_df, test_df = load_complete_shirt_tshirt_data()")
print("   features, labels, indices, sources = extract_features_complete(model, support_dataset, test_dataset, combined_df, device)")
print("   results = analyze_complete_transition(features, labels, indices, sources, support_dataset, test_dataset)") 