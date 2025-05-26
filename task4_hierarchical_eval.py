import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

def evaluate_hierarchical_classification(train_features, train_labels, test_features, test_labels, 
                                      category_names, k=1):
    """Evaluate classification performance using k-NN"""
    from sklearn.neighbors import KNeighborsClassifier
    
    # Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(train_features, train_labels)
    
    # Predict on test set
    predictions = knn.predict(test_features)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    
    # Generate detailed report
    report = classification_report(test_labels, predictions, 
                                 target_names=[category_names[i] for i in sorted(set(test_labels))],
                                 digits=3)
    
    print(f"\nClassification Accuracy: {accuracy:.3f}")
    print("\nDetailed Classification Report:")
    print(report)
    
    return predictions, accuracy

def visualize_confusion_matrix(true_labels, predictions, category_names):
    """Create and display confusion matrix"""
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[category_names[i] for i in sorted(set(true_labels))],
                yticklabels=[category_names[i] for i in sorted(set(true_labels))])
    plt.title('Confusion Matrix - Category Classification')
    plt.xlabel('Predicted Category')
    plt.ylabel('True Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def compare_hierarchical_performance(model, train_loader, test_loader, article_to_category, 
                                   category_names, device):
    """Compare performance between article-level and category-level classification"""
    # Extract features
    print("Extracting features...")
    train_features, train_article_labels = extract_features(model, train_loader, device)
    test_features, test_article_labels = extract_features(model, test_loader, device)
    
    # Convert to category labels
    train_category_labels = convert_to_category_labels(train_article_labels, article_to_category)
    test_category_labels = convert_to_category_labels(test_article_labels, article_to_category)
    
    # Evaluate article-level classification
    print("\nEvaluating article-level classification (39 classes)...")
    article_predictions, article_accuracy = evaluate_hierarchical_classification(
        train_features, train_article_labels, test_features, test_article_labels, 
        {i: f"Article_{i}" for i in range(39)})
    
    # Evaluate category-level classification
    print("\nEvaluating category-level classification (20 classes)...")
    category_predictions, category_accuracy = evaluate_hierarchical_classification(
        train_features, train_category_labels, test_features, test_category_labels, 
        category_names)
    
    # Visualize confusion matrix for categories
    visualize_confusion_matrix(test_category_labels, category_predictions, category_names)
    
    # Compare performances
    print("\nPerformance Comparison:")
    print(f"Article-level accuracy: {article_accuracy:.3f}")
    print(f"Category-level accuracy: {category_accuracy:.3f}")
    print(f"Difference (Category - Article): {category_accuracy - article_accuracy:.3f}")

if __name__ == "__main__":
    # Load the trained model from Task 2
    model = FashionNetT2(embedding_dim=128)  # Use your actual model class
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('model_task2.pth'))
    model.to(device)
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create category mapping
    article_to_category, category_names, article_names = create_article_to_category_mapping(train_dataset)
    
    # Run hierarchical evaluation
    compare_hierarchical_performance(model, train_loader, test_loader, 
                                   article_to_category, category_names, device) 