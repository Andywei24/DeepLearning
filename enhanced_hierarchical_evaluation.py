# Enhanced Step 4: Hierarchical Classification with Balanced Accuracy
# This version enhances the existing notebook implementation with balanced accuracy

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict

def create_article_to_category_mapping(dataset):
    """Create mapping from articleTypeId to categoryId using the dataset"""
    df = dataset.df
    mapping_df = df[['articleTypeId', 'categoryId', 'articleTypeName', 'categoryName']].drop_duplicates()
    mapping_df = mapping_df.sort_values('articleTypeId')
    
    # Create the mapping dictionary
    article_to_category = {}
    category_names = {}
    article_names = {}
    
    for _, row in mapping_df.iterrows():
        article_id = int(row['articleTypeId'])
        category_id = int(row['categoryId'])
        article_to_category[article_id] = category_id
        category_names[category_id] = row['categoryName']
        article_names[article_id] = row['articleTypeName']
    
    print("\nArticle Type to Category Mapping:")
    print("=" * 60)
    for article_id in sorted(article_to_category.keys()):
        category_id = article_to_category[article_id]
        print(f"Article {article_id:2d} ({article_names[article_id]:15s}) -> Category {category_id:2d} ({category_names[category_id]})")
    
    print(f"\nTotal Article Types: {len(article_to_category)}")
    print(f"Total Categories: {len(set(article_to_category.values()))}")
    
    return article_to_category, category_names, article_names

def extract_features(model, dataloader, device):
    """Extract features using the trained Task 2 model"""
    model.eval()
    features = []
    article_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # Get embeddings
            features.append(outputs.cpu().numpy())
            article_labels.extend(labels.numpy())
    
    features = np.vstack(features)
    article_labels = np.array(article_labels)
    
    return features, article_labels

def convert_to_category_labels(article_labels, article_to_category):
    """Convert article type labels to category labels"""
    return np.array([article_to_category[art_id] for art_id in article_labels])

def evaluate_hierarchical_classification_enhanced(train_features, train_labels, test_features, test_labels, 
                                                category_names, k=1, level_name=""):
    """Enhanced evaluation with both regular and balanced accuracy"""
    
    # Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(train_features, train_labels)
    
    # Predict on test set
    predictions = knn.predict(test_features)
    
    # Calculate both accuracies
    accuracy = accuracy_score(test_labels, predictions)
    balanced_acc = balanced_accuracy_score(test_labels, predictions)
    
    # Print enhanced results
    print(f"\n{level_name} Classification Results:")
    print("=" * 60)
    print(f"Regular Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Balanced Accuracy:    {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
    print(f"Number of samples:    {len(test_labels)}")
    print(f"Number of classes:    {len(np.unique(test_labels))}")
    
    # Calculate per-class metrics for insight
    unique_classes = np.unique(test_labels)
    per_class_acc = []
    class_counts = []
    
    for cls in unique_classes:
        mask = test_labels == cls
        cls_count = np.sum(mask)
        if cls_count > 0:
            cls_acc = accuracy_score(test_labels[mask], predictions[mask])
            per_class_acc.append(cls_acc)
            class_counts.append(cls_count)
    
    print(f"Mean per-class acc:   {np.mean(per_class_acc):.4f} ({np.mean(per_class_acc)*100:.2f}%)")
    print(f"Std per-class acc:    {np.std(per_class_acc):.4f}")
    print(f"Min/Max class size:   {min(class_counts)} / {max(class_counts)} samples")
    
    # Generate detailed report for smaller number of classes
    if len(unique_classes) <= 25:  # Only show detailed report for manageable number of classes
        try:
            target_names = [category_names.get(i, f"Class_{i}") for i in sorted(unique_classes)]
            report = classification_report(test_labels, predictions, 
                                         target_names=target_names,
                                         digits=3, zero_division=0)
            print(f"\nDetailed Classification Report:")
            print(report)
        except:
            print("\nDetailed report generation skipped due to class naming issues.")
    
    return predictions, accuracy, balanced_acc

def compare_hierarchical_performance_enhanced(model, train_loader, test_loader, article_to_category, 
                                            category_names, device):
    """Enhanced comparison with balanced accuracy metrics"""
    
    print("="*80)
    print("ENHANCED HIERARCHICAL CLASSIFICATION EVALUATION")
    print("Comparing Article-level vs Category-level with Balanced Accuracy")
    print("="*80)
    
    # Extract features
    print("Extracting features from training data...")
    train_features, train_article_labels = extract_features(model, train_loader, device)
    print("Extracting features from test data...")
    test_features, test_article_labels = extract_features(model, test_loader, device)
    
    # Convert to category labels
    train_category_labels = convert_to_category_labels(train_article_labels, article_to_category)
    test_category_labels = convert_to_category_labels(test_article_labels, article_to_category)
    
    # Evaluate article-level classification
    print("\n" + "="*70)
    print("ARTICLE-LEVEL CLASSIFICATION (Fine-grained)")
    print("="*70)
    article_predictions, article_accuracy, article_balanced_acc = evaluate_hierarchical_classification_enhanced(
        train_features, train_article_labels, test_features, test_article_labels, 
        {i: f"Article_{i}" for i in range(max(test_article_labels)+1)}, 
        level_name="Article-level")
    
    # Evaluate category-level classification
    print("\n" + "="*70)
    print("CATEGORY-LEVEL CLASSIFICATION (Coarse-grained)")
    print("="*70)
    category_predictions, category_accuracy, category_balanced_acc = evaluate_hierarchical_classification_enhanced(
        train_features, train_category_labels, test_features, test_category_labels, 
        category_names, level_name="Category-level")
    
    # Enhanced performance comparison
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"Article-level ({len(np.unique(test_article_labels))} classes):")
    print(f"  Regular Accuracy:     {article_accuracy:.4f} ({article_accuracy*100:.2f}%)")
    print(f"  Balanced Accuracy:    {article_balanced_acc:.4f} ({article_balanced_acc*100:.2f}%)")
    
    print(f"\nCategory-level ({len(np.unique(test_category_labels))} classes):")
    print(f"  Regular Accuracy:     {category_accuracy:.4f} ({category_accuracy*100:.2f}%)")
    print(f"  Balanced Accuracy:    {category_balanced_acc:.4f} ({category_balanced_acc*100:.2f}%)")
    
    # Calculate improvements
    improvement_regular = category_accuracy - article_accuracy
    improvement_balanced = category_balanced_acc - article_balanced_acc
    
    print(f"\nHierarchical Performance Gains:")
    print(f"  Regular Accuracy:     {improvement_regular:+.4f} ({improvement_regular*100:+.2f}%)")
    print(f"  Balanced Accuracy:    {improvement_balanced:+.4f} ({improvement_balanced*100:+.2f}%)")
    
    # Enhanced visualization
    create_enhanced_visualization(article_accuracy, article_balanced_acc, 
                                category_accuracy, category_balanced_acc,
                                len(np.unique(test_article_labels)), 
                                len(np.unique(test_category_labels)))
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION & INSIGHTS")
    print("="*80)
    
    if improvement_regular > 0 and improvement_balanced > 0:
        print("✅ EXCELLENT: Both metrics improve at category level!")
        print("   Your model learned meaningful hierarchical patterns.")
    elif improvement_regular > 0 or improvement_balanced > 0:
        print("✅ GOOD: At least one metric improves at category level.")
        if improvement_balanced > improvement_regular:
            print("   Balanced accuracy improvement suggests better class balance handling.")
        else:
            print("   Regular accuracy improvement suggests overall better performance.")
    else:
        print("⚠️  INTERESTING: Article-level performs better.")
        print("   Model learned very specific article-level discriminative features.")
    
    return {
        'article_accuracy': article_accuracy,
        'article_balanced_accuracy': article_balanced_acc,
        'category_accuracy': category_accuracy,
        'category_balanced_accuracy': category_balanced_acc,
        'improvements': {
            'regular': improvement_regular,
            'balanced': improvement_balanced
        }
    }

def create_enhanced_visualization(article_acc, article_bal_acc, category_acc, category_bal_acc,
                                num_article_classes, num_category_classes):
    """Create enhanced visualization with both accuracy metrics"""
    
    metrics = ['Regular Accuracy', 'Balanced Accuracy']
    article_scores = [article_acc, article_bal_acc]
    category_scores = [category_acc, category_bal_acc]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Main comparison plot
    bars1 = ax1.bar(x - width/2, article_scores, width, 
                   label=f'Article-level ({num_article_classes} classes)', 
                   alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, category_scores, width, 
                   label=f'Category-level ({num_category_classes} classes)', 
                   alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Accuracy Metric', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Enhanced Hierarchical Classification Performance\n(Task 2 Embedding Model + k-NN)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Improvement plot
    improvements = [category_acc - article_acc, category_bal_acc - article_bal_acc]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars3 = ax2.bar(metrics, improvements, alpha=0.7, color=colors)
    ax2.set_ylabel('Improvement (Category - Article)', fontsize=12)
    ax2.set_title('Performance Improvement\n(Category vs Article Level)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels for improvements
    for bar, imp in zip(bars3, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.005 if height > 0 else -0.015),
               f'{imp:+.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# Usage example for notebook:
"""
# Load the trained model from Task 2
model = FashionNetT2(embedding_dim=128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('task2_model.pth'))
model.to(device)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(main_test_dataset, batch_size=batch_size, shuffle=False)

# Create category mapping
article_to_category, category_names, article_names = create_article_to_category_mapping(train_dataset)

# Run enhanced hierarchical evaluation with balanced accuracy
results = compare_hierarchical_performance_enhanced(model, train_loader, test_loader, 
                                                  article_to_category, category_names, device)
""" 