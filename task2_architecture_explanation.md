# Task 2: Architecture & Implementation Explained

## 🏗️ Model Architecture Comparison

### **TASK 1 (Classification)**
```
Input Image (224x224x3)
       ↓
    Conv1 (32 filters) → ReLU → MaxPool
       ↓
    Conv2 (64 filters) → ReLU → MaxPool  
       ↓
    Conv3 (128 filters) → ReLU → MaxPool
       ↓
    Flatten → FC1 (512) → Dropout
       ↓
    FC2 (num_classes) ← OUTPUTS CLASS PROBABILITIES
       ↓
    Softmax → [0.1, 0.8, 0.05, 0.05] (probabilities for each class)
```

### **TASK 2 (Metric Learning)**
```
Input Image (224x224x3)
       ↓
    Conv1 (32 filters) → ReLU → MaxPool
       ↓
    Conv2 (64 filters) → ReLU → MaxPool
       ↓
    Conv3 (128 filters) → ReLU → MaxPool
       ↓
    Flatten → FC1 (512) → Dropout
       ↓
    Embedding (128) ← OUTPUTS FEATURE EMBEDDINGS
       ↓
    L2 Normalize → [0.2, -0.1, 0.3, ...] (128-dim normalized vector)
```

## 🎯 Key Difference
- **Task 1**: Outputs class probabilities → "This is 80% likely to be a shirt"
- **Task 2**: Outputs feature embeddings → "This item has features [0.2, -0.1, 0.3, ...]"

---

## 🔄 Complete Workflow

### **PHASE 1: TRAINING (Metric Learning)**

```
1. Take a batch of training images
2. Get embeddings for all images
3. For each image, find:
   - POSITIVE: Another image from SAME class
   - NEGATIVE: An image from DIFFERENT class
4. Apply Triplet Loss: make (anchor, positive) closer and (anchor, negative) farther
5. Repeat until model learns good embeddings
```

**Example:**
```
Anchor: Shirt embedding    = [0.2, 0.5, -0.1, ...]
Positive: Another shirt    = [0.3, 0.4, -0.2, ...]  ← Should be CLOSE
Negative: Pants            = [-0.8, 0.1, 0.9, ...]  ← Should be FAR

Triplet Loss = max(0, distance(anchor,positive) - distance(anchor,negative) + margin)
```

### **PHASE 2: FEW-SHOT CLASSIFICATION**

```
1. SUPPORT SET: Create prototypes
   - For each class in support set, calculate MEAN embedding
   - Shirt prototype = mean of all shirt embeddings in support set
   
2. TEST CLASSIFICATION: 
   - Get embedding for test image
   - Find which prototype is CLOSEST
   - Assign that class label
```

**Example:**
```
Support Set:
- Class "Dress": [emb1, emb2, emb3] → Prototype = mean([emb1, emb2, emb3])
- Class "Shoes": [emb4, emb5] → Prototype = mean([emb4, emb5])

Test Image: embedding = [0.1, 0.3, -0.2, ...]
- Distance to Dress prototype: 0.4
- Distance to Shoes prototype: 0.8
→ Prediction: "Dress" (closer distance)
```

---

## 🧠 Techniques Used

### **1. Metric Learning (Triplet Loss)**
**Purpose**: Learn embeddings where similar items are close, different items are far

**How it works**:
```python
# Simple triplet loss
pos_dist = distance(anchor, positive)    # Distance to same class
neg_dist = distance(anchor, negative)    # Distance to different class
loss = max(0, pos_dist - neg_dist + margin)  # Want pos_dist < neg_dist
```

**Why this works**: 
- Pulls shirts closer to other shirts
- Pushes shirts away from pants
- Creates meaningful embedding space

### **2. Prototype-Based Classification**
**Purpose**: Classify without retraining using support set

**How it works**:
```python
# Create prototypes (class representatives)
for each class in support_set:
    prototype[class] = mean(all_embeddings_of_this_class)

# Classify test image
test_embedding = model(test_image)
predicted_class = closest_prototype(test_embedding)
```

**Why this works**:
- No retraining needed
- Works with any number of classes
- Uses "average" representation of each class

### **3. L2 Normalization**
**Purpose**: Make all embeddings have same length for fair distance comparison

**How it works**:
```python
embedding = [2, 4, 6]  # Original
normalized = [0.27, 0.54, 0.81]  # After L2 norm (length = 1)
```

**Why this works**:
- All embeddings on unit sphere
- Distance reflects similarity, not magnitude
- Stable training and inference

---

## 📊 Evaluation Process

### **The 4 Scenarios**

```
Scenario 1: Use TRAINING DATA as support set
           → Easy (model has seen these exact images)
           
Scenario 2: Use SEPARATE SUPPORT SET for main classes  
           → Medium (same classes, different images)
           
Scenario 3: Use NEW CLASSES support set
           → Hard (completely unseen classes) ← KEY TEST
           
Scenario 4: Use MIXED support set (main + new classes)
           → Medium-Hard (combination)
```

**Evaluation Flow**:
```
For each scenario:
1. Load support_dataset and test_dataset
2. Create prototypes from support_dataset  
3. Classify test_dataset using prototypes
4. Calculate accuracy and balanced_accuracy
5. Check if Scenario 3 ≥ 45% (target)
```

---

## 💡 Why This Approach Works for Fashion Trends

**Problem**: New fashion items appear that weren't in training data

**Traditional Solution**: Retrain entire model (expensive, slow)

**Our Solution**: 
1. Learn general fashion features during training
2. When new trend appears, just show a few examples (support set)
3. Create prototype for new trend
4. Classify using distance to prototypes

**Example**:
```
Training: Learned features for [shirts, pants, dresses, shoes]
New Trend: "Crop tops" appear
Solution: 
- Show model 5 crop top examples (support set)
- Create crop top prototype
- Now can classify any crop top using prototype matching
- NO RETRAINING NEEDED!
```

---

## 🎯 Key Benefits

1. **Scalable**: Add new classes without retraining
2. **Efficient**: Fast inference using prototype matching  
3. **Flexible**: Works with few examples per new class
4. **Generalizable**: Learned features transfer to new domains

This is why it's called "few-shot learning" - you need only a FEW examples to learn a new class! 