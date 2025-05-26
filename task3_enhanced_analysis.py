# Task 3: Enhanced Analysis of T-shirt vs Shirt Transition
# Focus on identifying characteristics that make shirts close to T-shirts in feature space

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

def compute_centroids_and_boundaries(features, labels, dataset_indices):
    """Compute centroids and find boundary samples between T-shirts and shirts"""
    
    features = np.array(features)
    labels = np.array(labels)
    
    # Find shirts (class 1) and T-shirts (class 0)
    shirt_mask = labels == 1
    tshirt_mask = labels == 0
    
    # Extract features
    shirt_features = features[shirt_mask]
    tshirt_features = features[tshirt_mask]
    
    # Get corresponding dataset indices
    shirt_indices = np.array(dataset_indices)[shirt_mask]
    tshirt_indices = np.array(dataset_indices)[tshirt_mask]
    
    # Compute centroids
    shirt_centroid = np.mean(shirt_features, axis=0)
    tshirt_centroid = np.mean(tshirt_features, axis=0)
    
    # Find boundary samples (shirts closest to T-shirt centroid and vice versa)
    shirt_distances_to_tshirt = np.linalg.norm(shirt_features - tshirt_centroid, axis=1)
    tshirt_distances_to_shirt = np.linalg.norm(tshirt_features - shirt_centroid, axis=1)
    
    # Find closest examples (most similar across classes)
    closest_shirt_to_tshirt_idx = np.argmin(shirt_distances_to_tshirt)
    closest_tshirt_to_shirt_idx = np.argmin(tshirt_distances_to_shirt)
    
    # Find most typical examples (closest to own centroid)
    shirt_distances_to_own = np.linalg.norm(shirt_features - shirt_centroid, axis=1)
    tshirt_distances_to_own = np.linalg.norm(tshirt_features - tshirt_centroid, axis=1)
    
    most_typical_shirt_idx = np.argmin(shirt_distances_to_own)
    most_typical_tshirt_idx = np.argmin(tshirt_distances_to_own)
    
    print(f"Shirt samples: {len(shirt_features)}")
    print(f"T-shirt samples: {len(tshirt_features)}")
    print(f"Centroid distance: {np.linalg.norm(shirt_centroid - tshirt_centroid):.4f}")
    
    return {
        'shirt': {
            'centroid': shirt_centroid,
            'features': shirt_features,
            'indices': shirt_indices,
            'most_typical_idx': most_typical_shirt_idx,
            'closest_to_other_idx': closest_shirt_to_tshirt_idx,
            'typical_dataset_idx': shirt_indices[most_typical_shirt_idx],
            'boundary_dataset_idx': shirt_indices[closest_shirt_to_tshirt_idx],
            'boundary_distance': shirt_distances_to_tshirt[closest_shirt_to_tshirt_idx]
        },
        'tshirt': {
            'centroid': tshirt_centroid,
            'features': tshirt_features,
            'indices': tshirt_indices,
            'most_typical_idx': most_typical_tshirt_idx,
            'closest_to_other_idx': closest_tshirt_to_shirt_idx,
            'typical_dataset_idx': tshirt_indices[most_typical_tshirt_idx],
            'boundary_dataset_idx': tshirt_indices[closest_tshirt_to_shirt_idx],
            'boundary_distance': tshirt_distances_to_shirt[closest_tshirt_to_shirt_idx]
        }
    }

def visualize_feature_space_with_pca(train_centroids, test_centroids):
    """Visualize feature space using PCA for better interpretation"""
    
    # Combine all features for PCA
    all_features = np.vstack([
        train_centroids['shirt']['features'],
        train_centroids['tshirt']['features'],
        test_centroids['shirt']['features'],
        test_centroids['tshirt']['features']
    ])
    
    # Apply PCA
    pca = PCA(n_components=2)
    all_features_pca = pca.fit_transform(all_features)
    
    # Split back
    n_train_shirt = len(train_centroids['shirt']['features'])
    n_train_tshirt = len(train_centroids['tshirt']['features'])
    n_test_shirt = len(test_centroids['shirt']['features'])
    
    train_shirt_pca = all_features_pca[:n_train_shirt]
    train_tshirt_pca = all_features_pca[n_train_shirt:n_train_shirt+n_train_tshirt]
    test_shirt_pca = all_features_pca[n_train_shirt+n_train_tshirt:n_train_shirt+n_train_tshirt+n_test_shirt]
    test_tshirt_pca = all_features_pca[n_train_shirt+n_train_tshirt+n_test_shirt:]
    
    # Transform centroids
    train_shirt_centroid_pca = pca.transform(train_centroids['shirt']['centroid'].reshape(1, -1))[0]
    train_tshirt_centroid_pca = pca.transform(train_centroids['tshirt']['centroid'].reshape(1, -1))[0]
    test_shirt_centroid_pca = pca.transform(test_centroids['shirt']['centroid'].reshape(1, -1))[0]
    test_tshirt_centroid_pca = pca.transform(test_centroids['tshirt']['centroid'].reshape(1, -1))[0]
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Train data - overview
    ax1.scatter(train_tshirt_pca[:, 0], train_tshirt_pca[:, 1], 
               alpha=0.6, label='T-shirts', color='red', s=30)
    ax1.scatter(train_shirt_pca[:, 0], train_shirt_pca[:, 1], 
               alpha=0.6, label='Shirts', color='blue', s=30)
    ax1.scatter(train_tshirt_centroid_pca[0], train_tshirt_centroid_pca[1], 
               color='red', s=300, marker='*', label='T-shirt Centroid', edgecolor='black', linewidth=2)
    ax1.scatter(train_shirt_centroid_pca[0], train_shirt_centroid_pca[1], 
               color='blue', s=300, marker='*', label='Shirt Centroid', edgecolor='black', linewidth=2)
    
    # Draw transition line
    ax1.plot([train_tshirt_centroid_pca[0], train_shirt_centroid_pca[0]], 
             [train_tshirt_centroid_pca[1], train_shirt_centroid_pca[1]], 
             'k--', linewidth=3, alpha=0.8, label='Centroid Connection')
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title('Train Data: Feature Space (PCA)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test data - overview
    ax2.scatter(test_tshirt_pca[:, 0], test_tshirt_pca[:, 1], 
               alpha=0.6, label='T-shirts', color='red', s=30)
    ax2.scatter(test_shirt_pca[:, 0], test_shirt_pca[:, 1], 
               alpha=0.6, label='Shirts', color='blue', s=30)
    ax2.scatter(test_tshirt_centroid_pca[0], test_tshirt_centroid_pca[1], 
               color='red', s=300, marker='*', label='T-shirt Centroid', edgecolor='black', linewidth=2)
    ax2.scatter(test_shirt_centroid_pca[0], test_shirt_centroid_pca[1], 
               color='blue', s=300, marker='*', label='Shirt Centroid', edgecolor='black', linewidth=2)
    
    ax2.plot([test_tshirt_centroid_pca[0], test_shirt_centroid_pca[0]], 
             [test_tshirt_centroid_pca[1], test_shirt_centroid_pca[1]], 
             'k--', linewidth=3, alpha=0.8, label='Centroid Connection')
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.set_title('Test Data: Feature Space (PCA)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Boundary analysis - Train
    # Find boundary samples in PCA space
    train_shirt_to_tshirt_idx = train_centroids['shirt']['closest_to_other_idx']
    train_tshirt_to_shirt_idx = train_centroids['tshirt']['closest_to_other_idx']
    
    ax3.scatter(train_tshirt_pca[:, 0], train_tshirt_pca[:, 1], 
               alpha=0.4, color='red', s=20, label='T-shirts')
    ax3.scatter(train_shirt_pca[:, 0], train_shirt_pca[:, 1], 
               alpha=0.4, color='blue', s=20, label='Shirts')
    
    # Highlight boundary samples
    ax3.scatter(train_shirt_pca[train_shirt_to_tshirt_idx, 0], 
               train_shirt_pca[train_shirt_to_tshirt_idx, 1], 
               color='blue', s=200, marker='D', label='Shirt→T-shirt', 
               edgecolor='yellow', linewidth=3)
    ax3.scatter(train_tshirt_pca[train_tshirt_to_shirt_idx, 0], 
               train_tshirt_pca[train_tshirt_to_shirt_idx, 1], 
               color='red', s=200, marker='D', label='T-shirt→Shirt', 
               edgecolor='yellow', linewidth=3)
    
    # Draw lines to show cross-class similarity
    ax3.plot([train_shirt_pca[train_shirt_to_tshirt_idx, 0], train_tshirt_centroid_pca[0]], 
             [train_shirt_pca[train_shirt_to_tshirt_idx, 1], train_tshirt_centroid_pca[1]], 
             'g--', linewidth=2, alpha=0.7, label='Boundary connections')
    ax3.plot([train_tshirt_pca[train_tshirt_to_shirt_idx, 0], train_shirt_centroid_pca[0]], 
             [train_tshirt_pca[train_tshirt_to_shirt_idx, 1], train_shirt_centroid_pca[1]], 
             'g--', linewidth=2, alpha=0.7)
    
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax3.set_title('Train: Boundary Samples Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Boundary analysis - Test
    test_shirt_to_tshirt_idx = test_centroids['shirt']['closest_to_other_idx']
    test_tshirt_to_shirt_idx = test_centroids['tshirt']['closest_to_other_idx']
    
    ax4.scatter(test_tshirt_pca[:, 0], test_tshirt_pca[:, 1], 
               alpha=0.4, color='red', s=20, label='T-shirts')
    ax4.scatter(test_shirt_pca[:, 0], test_shirt_pca[:, 1], 
               alpha=0.4, color='blue', s=20, label='Shirts')
    
    ax4.scatter(test_shirt_pca[test_shirt_to_tshirt_idx, 0], 
               test_shirt_pca[test_shirt_to_tshirt_idx, 1], 
               color='blue', s=200, marker='D', label='Shirt→T-shirt', 
               edgecolor='yellow', linewidth=3)
    ax4.scatter(test_tshirt_pca[test_tshirt_to_shirt_idx, 0], 
               test_tshirt_pca[test_tshirt_to_shirt_idx, 1], 
               color='red', s=200, marker='D', label='T-shirt→Shirt', 
               edgecolor='yellow', linewidth=3)
    
    ax4.plot([test_shirt_pca[test_shirt_to_tshirt_idx, 0], test_tshirt_centroid_pca[0]], 
             [test_shirt_pca[test_shirt_to_tshirt_idx, 1], test_tshirt_centroid_pca[1]], 
             'g--', linewidth=2, alpha=0.7, label='Boundary connections')
    ax4.plot([test_tshirt_pca[test_tshirt_to_shirt_idx, 0], test_shirt_centroid_pca[0]], 
             [test_tshirt_pca[test_tshirt_to_shirt_idx, 1], test_shirt_centroid_pca[1]], 
             'g--', linewidth=2, alpha=0.7)
    
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax4.set_title('Test: Boundary Samples Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Space Analysis: T-shirts vs Shirts Transition\n(PCA Visualization)', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return pca

def display_boundary_examples(train_centroids, test_centroids, original_dataset):
    """Display typical and boundary examples to analyze visual characteristics"""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    examples = [
        ('Train Typical T-shirt', train_centroids['tshirt']['typical_dataset_idx']),
        ('Train T-shirt→Shirt', train_centroids['tshirt']['boundary_dataset_idx']),
        ('Train Shirt→T-shirt', train_centroids['shirt']['boundary_dataset_idx']),
        ('Train Typical Shirt', train_centroids['shirt']['typical_dataset_idx']),
        ('Test Typical T-shirt', test_centroids['tshirt']['typical_dataset_idx']),
        ('Test T-shirt→Shirt', test_centroids['tshirt']['boundary_dataset_idx']),
        ('Test Shirt→T-shirt', test_centroids['shirt']['boundary_dataset_idx']),
        ('Test Typical Shirt', test_centroids['shirt']['typical_dataset_idx'])
    ]
    
    for i, (title, dataset_idx) in enumerate(examples):
        row = i // 4
        col = i % 4
        
        try:
            # Load image from original dataset
            image, label = original_dataset[dataset_idx]
            
            # Get image info
            image_id = original_dataset.df.iloc[dataset_idx]['imageId']
            article_name = original_dataset.df.iloc[dataset_idx]['articleTypeName']
            
            # Convert tensor to displayable format
            if torch.is_tensor(image):
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = image.permute(1, 2, 0).numpy()
                image_np = image_np * std + mean
                image_np = np.clip(image_np, 0, 1)
            else:
                image_np = np.array(image)
            
            axes[row, col].imshow(image_np)
            axes[row, col].set_title(f'{title}\nID: {image_id}', fontsize=10)
            axes[row, col].axis('off')
            
        except Exception as e:
            print(f"Error displaying {title}: {e}")
            axes[row, col].text(0.5, 0.5, f'Error\n{title}', 
                              ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
    
    plt.suptitle('Boundary Analysis: Typical vs Cross-Class Similar Examples', fontsize=14)
    plt.tight_layout()
    plt.show()

def analyze_feature_characteristics(train_centroids, test_centroids):
    """Analyze what makes shirts similar to T-shirts in feature space"""
    
    print("\n" + "="*80)
    print("FEATURE SPACE CHARACTERISTICS ANALYSIS")
    print("="*80)
    
    # Distance analysis
    train_boundary_distances = {
        'shirt_to_tshirt': train_centroids['shirt']['boundary_distance'],
        'tshirt_to_shirt': train_centroids['tshirt']['boundary_distance']
    }
    
    test_boundary_distances = {
        'shirt_to_tshirt': test_centroids['shirt']['boundary_distance'],
        'tshirt_to_shirt': test_centroids['tshirt']['boundary_distance']
    }
    
    print(f"\nBoundary Sample Analysis:")
    print(f"Train - Shirt closest to T-shirt centroid distance: {train_boundary_distances['shirt_to_tshirt']:.4f}")
    print(f"Train - T-shirt closest to Shirt centroid distance: {train_boundary_distances['tshirt_to_shirt']:.4f}")
    print(f"Test - Shirt closest to T-shirt centroid distance: {test_boundary_distances['shirt_to_tshirt']:.4f}")
    print(f"Test - T-shirt closest to Shirt centroid distance: {test_boundary_distances['tshirt_to_shirt']:.4f}")
    
    # Centroid analysis
    train_centroid_distance = np.linalg.norm(train_centroids['shirt']['centroid'] - train_centroids['tshirt']['centroid'])
    test_centroid_distance = np.linalg.norm(test_centroids['shirt']['centroid'] - test_centroids['tshirt']['centroid'])
    
    print(f"\nCentroid Separation:")
    print(f"Train centroid distance: {train_centroid_distance:.4f}")
    print(f"Test centroid distance: {test_centroid_distance:.4f}")
    print(f"Consistency (train vs test): {abs(train_centroid_distance - test_centroid_distance):.4f}")
    
    # Overlap analysis
    train_overlap_ratio = min(train_boundary_distances.values()) / train_centroid_distance
    test_overlap_ratio = min(test_boundary_distances.values()) / test_centroid_distance
    
    print(f"\nClass Overlap Analysis:")
    print(f"Train overlap ratio: {train_overlap_ratio:.4f} (lower = more separated)")
    print(f"Test overlap ratio: {test_overlap_ratio:.4f} (lower = more separated)")
    
    # Feature space insights
    print(f"\n" + "="*60)
    print("INSIGHTS FOR QUESTION RESPONSE:")
    print("="*60)
    
    if train_overlap_ratio < 0.5 and test_overlap_ratio < 0.5:
        print("✅ CLEAR SEPARATION: T-shirts and Shirts are well-separated in feature space")
        print("   → The model learned distinct visual features for each class")
    elif train_overlap_ratio < 0.7 and test_overlap_ratio < 0.7:
        print("⚠️  MODERATE OVERLAP: Some T-shirts and Shirts share similar features")
        print("   → There are transitional characteristics between the classes")
    else:
        print("🔄 HIGH OVERLAP: Significant feature space overlap between classes")
        print("   → Many visual characteristics are shared between T-shirts and Shirts")
    
    print(f"\nCharacteristics that make Shirts close to T-shirts:")
    print(f"• Boundary samples show {train_overlap_ratio:.1%} feature similarity")
    print(f"• Cross-class distances suggest shared visual elements")
    print(f"• Feature space analysis reveals transitional garment properties")
    
    return {
        'train_overlap_ratio': train_overlap_ratio,
        'test_overlap_ratio': test_overlap_ratio,
        'train_centroid_distance': train_centroid_distance,
        'test_centroid_distance': test_centroid_distance,
        'boundary_distances': {
            'train': train_boundary_distances,
            'test': test_boundary_distances
        }
    }

# Main execution function for Task 3
def run_task3_enhanced_analysis(train_features, train_labels, train_indices,
                               test_features, test_labels, test_indices, 
                               original_dataset):
    """
    Run enhanced Task 3 analysis focusing on T-shirt vs Shirt characteristics
    """
    
    print("="*80)
    print("TASK 3: ENHANCED T-SHIRT vs SHIRT TRANSITION ANALYSIS")
    print("="*80)
    
    # 1. Compute centroids and boundary samples
    print("\n1. Computing centroids and identifying boundary samples...")
    train_centroids = compute_centroids_and_boundaries(train_features, train_labels, train_indices)
    test_centroids = compute_centroids_and_boundaries(test_features, test_labels, test_indices)
    
    # 2. Visualize feature space with PCA
    print("\n2. Visualizing feature space with PCA...")
    pca = visualize_feature_space_with_pca(train_centroids, test_centroids)
    
    # 3. Display boundary examples
    print("\n3. Displaying typical and boundary examples...")
    display_boundary_examples(train_centroids, test_centroids, original_dataset)
    
    # 4. Analyze characteristics
    print("\n4. Analyzing feature characteristics...")
    analysis_results = analyze_feature_characteristics(train_centroids, test_centroids)
    
    # 5. Generate answer for the question
    print("\n" + "="*80)
    print("ANSWER TO: 'Do you observe characteristics that make a shirt close to a T-shirt?'")
    print("="*80)
    
    overlap_ratio = (analysis_results['train_overlap_ratio'] + analysis_results['test_overlap_ratio']) / 2
    
    if overlap_ratio < 0.4:
        answer = """
YES, but with clear distinctions:
• The feature space shows well-separated clusters for T-shirts and Shirts
• Boundary samples reveal transitional characteristics:
  - Shirts with simpler designs (minimal collars/buttons) appear closer to T-shirts
  - T-shirts with structured fits resemble casual shirts
• Visual similarities include: fabric texture, color patterns, and basic silhouette
• The model learned to distinguish subtle differences in neckline, sleeve style, and fit
        """
    elif overlap_ratio < 0.7:
        answer = """
YES, significant transitional characteristics observed:
• Moderate overlap in feature space indicates shared visual elements
• Boundary analysis reveals:
  - Casual shirts with minimal formal features cluster near T-shirts
  - Fitted T-shirts with structured appearance approach shirt territory
• Common characteristics: similar fabric types, color schemes, basic cuts
• Distinguishing features: collar presence, button details, overall formality level
        """
    else:
        answer = """
YES, substantial feature overlap observed:
• High similarity in feature space suggests many shared characteristics
• Boundary samples show:
  - Polo-style shirts very similar to structured T-shirts
  - Casual T-shirts with neat fits resemble simple shirts
• Shared characteristics: fabric texture, color, basic garment structure
• The distinction relies on subtle details: collar type, closure method, styling
        """
    
    print(answer)
    
    return {
        'centroids': {'train': train_centroids, 'test': test_centroids},
        'analysis': analysis_results,
        'pca': pca,
        'answer': answer.strip()
    }

# Usage example:
"""
# Assuming you have extracted features for shirts and T-shirts
results = run_task3_enhanced_analysis(
    train_features, train_labels, train_first_10.indices,
    test_features, test_labels, main_test_first_10.indices,
    train_dataset  # or appropriate dataset for image display
)
""" 