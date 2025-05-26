# Calculate Regular and Balanced Accuracy
# Run this cell after you have your model and data loaders ready

from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Load your trained model (modify path as needed)
model = FashionNetT2(embedding_dim=128)  # Use your actual model class
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('task2_model.pth'))  # Update path if needed
model.to(device)

# Create data loaders if not already created
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(main_test_dataset, batch_size=batch_size, shuffle=False)

# Create category mapping if not already done
article_to_category, category_names, article_names = create_article_to_category_mapping(train_dataset)

# Run the comprehensive accuracy evaluation
print("Starting accuracy calculation...")
results = run_accuracy_evaluation(model, train_loader, test_loader, 
                                 article_to_category, category_names, article_names, device)

# Extract key results
article_acc = results['article_results']['accuracy']
article_balanced_acc = results['article_results']['balanced_accuracy']
category_acc = results['category_results']['accuracy']
category_balanced_acc = results['category_results']['balanced_accuracy']

print("\n" + "="*80)
print("FINAL ACCURACY SUMMARY")
print("="*80)
print(f"Article-level (39 classes):")
print(f"  Regular Accuracy:    {article_acc:.4f} ({article_acc*100:.2f}%)")
print(f"  Balanced Accuracy:   {article_balanced_acc:.4f} ({article_balanced_acc*100:.2f}%)")
print(f"\nCategory-level (20 classes):")
print(f"  Regular Accuracy:    {category_acc:.4f} ({category_acc*100:.2f}%)")
print(f"  Balanced Accuracy:   {category_balanced_acc:.4f} ({category_balanced_acc*100:.2f}%)")
print("="*80) 