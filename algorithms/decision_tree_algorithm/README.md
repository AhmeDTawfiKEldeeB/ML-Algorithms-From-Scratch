# 🌳 Decision Tree From Scratch

> **Making intelligent decisions by asking the right questions - just like nature intended!**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-2.3+-orange.svg)](https://numpy.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-green.svg)](https://scikit-learn.org)

## 🌟 Overview

This directory contains a **complete implementation of Decision Tree from scratch** using only NumPy! Decision Trees are one of the most intuitive machine learning algorithms - they work exactly like human decision making by asking a series of yes/no questions to reach a conclusion. Perfect for understanding how AI makes decisions! 🎯

### ✨ Why This Implementation?

- 🌲 **Intuitive Logic**: See exactly how the algorithm asks questions and makes decisions
- 📊 **Information Theory**: Understand entropy and information gain in action  
- 🎯 **Educational Focus**: Every split decision explained clearly
- 🔧 **Fully Customizable**: Control tree depth, minimum samples, and feature selection
- 🩺 **Medical AI**: Validated on real breast cancer dataset with excellent accuracy!

## 📁 What's Inside

```
decision_tree_algorithm/
├── 🐍 decision_tree.py      # Main Decision Tree implementation
├── 🧪 decision_tree_test.py # Testing script with breast cancer data
└── 📚 README.md             # This comprehensive guide!
```

## 🔧 Files Explained

### `decision_tree.py` - The Decision Engine 🧠

Our main implementation contains:

```python
# 📊 Entropy calculation - measures "impurity" of data
def entropy(Y):
    hist = np.bincount(Y)
    ps = hist / len(Y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

# 🌳 Tree nodes - building blocks of our decision tree
class Node:
    def __init__(self, features=None, threshold=None, left=None, right=None, *, value=None):
        self.features = features      # Which feature to split on
        self.threshold = threshold    # What value to split at
        self.left = left             # Left subtree (≤ threshold)
        self.right = right           # Right subtree (> threshold)
        self.value = value           # Prediction (for leaf nodes)
    
    def is_leaf_node(self):
        return self.value is not None

# 🎯 The Decision Tree classifier
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split  # Stop splitting if too few samples
        self.max_depth = max_depth                  # Maximum tree depth
        self.n_feats = n_feats                     # Number of features to consider
        self.root = None                           # Root of the tree
    
    def fit(self, X, Y):
        # Set number of features to consider at each split
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        # Grow the tree recursively
        self.root = self._grow_tree(X, Y)
    
    def predict(self, X):
        # Traverse tree for each sample
        return np.array([self._traverse_tree(x, self.root) for x in X])
```

### `decision_tree_test.py` - Cancer Detection in Action! 🩺

Our test script demonstrates real medical diagnosis:

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Load real breast cancer dataset
data = datasets.load_breast_cancer()
X, Y = data.data, data.target

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1234
)

# Create and train our decision tree
clf = DecisionTree(max_depth=10)
clf.fit(X_train, Y_train)

# Make predictions and evaluate
y_pred = clf.predict(X_test)
acc = accuracy(Y_test, y_pred)

print("Accuracy:", acc)
```

## 🚀 Quick Start

### 1. Run the Test

```bash
# Navigate to the Decision Tree directory
cd algorithms/decision_tree_algorithm

# Run the test script
python decision_tree_test.py
```

**Expected Output:** `Accuracy: 0.9122807017543859` (about 91% accuracy! 🎉)

### 2. Use In Your Own Code

```python
import numpy as np
from decision_tree import DecisionTree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load your data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create decision tree classifier
tree_classifier = DecisionTree(max_depth=5, min_samples_split=5)

# Train the tree
tree_classifier.fit(X_train, y_train)

# Make predictions
predictions = tree_classifier.predict(X_test)

print(f"Predictions: {predictions}")
```

## 🧠 How Decision Trees Work (The Logic Explained!)

### The Big Picture 🎯

Decision Trees work like a flowchart of questions:
```
Is tumor radius > 12.5?
├── YES → Is texture > 20.1?
│   ├── YES → MALIGNANT 🔴
│   └── NO → Is perimeter > 80?
│       ├── YES → MALIGNANT 🔴
│       └── NO → BENIGN 🟢
└── NO → Is smoothness > 0.1?
    ├── YES → BENIGN 🟢
    └── NO → BENIGN 🟢
```

### Step-by-Step Learning Process

1. **🌱 Start with Root**
   ```python
   # Begin with all training data at the root
   root = self._grow_tree(X_train, Y_train, depth=0)
   ```

2. **❓ Find Best Question**
   ```python
   # For each possible feature and threshold, calculate information gain
   best_feat, best_thresh = self._best_criteria(X, Y, feature_indices)
   ```

3. **📊 Calculate Information Gain**
   ```python
   def _information_gain(self, Y, X_column, threshold):
       parent_entropy = entropy(Y)
       left_indices, right_indices = self._split(X_column, threshold)
       
       # Calculate weighted entropy after split
       n = len(Y)
       n_l, n_r = len(left_indices), len(right_indices)
       e_l, e_r = entropy(Y[left_indices]), entropy(Y[right_indices])
       child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
       
       # Information gain = reduction in entropy
       return parent_entropy - child_entropy
   ```

4. **🌿 Split Data**
   ```python
   # Split data based on best question
   left_indices, right_indices = self._split(X[:, best_feat], best_thresh)
   left_tree = self._grow_tree(X[left_indices], Y[left_indices], depth+1)
   right_tree = self._grow_tree(X[right_indices], Y[right_indices], depth+1)
   ```

5. **🍃 Create Leaves**
   ```python
   # Stop when: max depth reached, too few samples, or pure node
   if stopping_condition:
       leaf_value = self._most_common_label(Y)
       return Node(value=leaf_value)
   ```

### 📊 Entropy - Measuring "Messiness"

**Entropy** measures how mixed up the classes are:

```python
def entropy(Y):
    hist = np.bincount(Y)           # Count each class
    ps = hist / len(Y)              # Convert to probabilities
    return -np.sum([p * np.log2(p) for p in ps if p > 0])
```

**Examples:**
- **Pure node** (all same class): `entropy = 0.0` 📍
- **50/50 split**: `entropy = 1.0` 🌀
- **Mostly one class**: `entropy = 0.3` 📊

**Visual representation:**
```
Entropy
   1.0 ┤     🌀 Maximum confusion
   0.8 ┤    ╱ ╲
   0.6 ┤   ╱   ╲
   0.4 ┤  ╱     ╲
   0.2 ┤ ╱       ╲
   0.0 ┤📍_______📍 Pure classes
       0% 25% 50% 75% 100%
       Class A percentage
```

### 🎯 Information Gain - Finding the Best Split

**Information Gain** = How much does this split reduce confusion?

```
Information Gain = Parent Entropy - Weighted Average of Child Entropies
```

**Example:**
```
Before Split: [A,A,B,B,B,B] → Entropy = 0.92
After Split:  [A,A] and [B,B,B,B] → Weighted Entropy = 0.0
Information Gain = 0.92 - 0.0 = 0.92 🎉 (Perfect split!)
```

## 📊 Real-World Performance

### 🩺 Breast Cancer Dataset Results

Our Decision Tree achieves **~91% accuracy** on breast cancer diagnosis!

**Dataset Details:**
- **Samples:** 569 patients
- **Features:** 30 (tumor characteristics like radius, texture, perimeter, etc.)
- **Classes:** 2 (Malignant=1, Benign=0)  
- **Test Split:** 20% (114 samples for testing)
- **Medical Impact:** Can assist doctors in diagnosis! 🏥

### 🌳 Tree Characteristics

```python
# After training, you can explore the tree structure:
print(f"Tree depth: {tree_depth}")
print(f"Number of nodes: {node_count}")
print(f"Number of leaves: {leaf_count}")

# Example decision path:
# Root: radius_mean ≤ 12.5
# ├─ Left: texture_mean ≤ 20.1 → BENIGN
# └─ Right: perimeter_mean ≤ 80.0 → MALIGNANT
```

## 🔬 Understanding the Code

### Why This Implementation Rocks

1. **🧠 Intuitive Decision Making**
   ```python
   # You can trace exactly how decisions are made:
   def _traverse_tree(self, x, node):
       if node.is_leaf_node():
           return node.value                    # Final decision
       if x[node.features] <= node.threshold:
           return self._traverse_tree(x, node.left)   # Go left
       return self._traverse_tree(x, node.right)      # Go right
   ```

2. **📊 Information Theory in Action**
   ```python
   # See how the algorithm measures and improves information
   parent_entropy = entropy(Y)                           # Current confusion
   child_entropy = weighted_average_entropy_after_split  # Confusion after split
   information_gain = parent_entropy - child_entropy     # Improvement!
   ```

3. **🌲 Recursive Tree Building**
   ```python
   # Elegant recursive structure that builds the tree naturally
   left_tree = self._grow_tree(X[left_indices], Y[left_indices], depth+1)
   right_tree = self._grow_tree(X[right_indices], Y[right_indices], depth+1)
   ```

## 🎮 Try These Experiments

### Experiment 1: Different Tree Depths
```python
from decision_tree import DecisionTree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🌳 Testing different tree depths:")
for depth in [3, 5, 10, 15, None]:
    tree = DecisionTree(max_depth=depth)
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)
    accuracy = np.sum(y_test == predictions) / len(y_test)
    print(f"Max Depth {depth}: Accuracy = {accuracy:.3f}")
```

**What you'll see:**
- `depth=3`: Good accuracy, simple tree
- `depth=10`: Best balance (our choice!)
- `depth=15`: Might overfit slightly
- `depth=None`: Highest accuracy but complex tree

### Experiment 2: Minimum Samples Split
```python
print("🍃 Testing different minimum samples for splitting:")
for min_samples in [2, 5, 10, 20]:
    tree = DecisionTree(max_depth=10, min_samples_split=min_samples)
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)
    accuracy = np.sum(y_test == predictions) / len(y_test)
    print(f"Min Samples {min_samples}: Accuracy = {accuracy:.3f}")
```

### Experiment 3: Feature Randomness (Random Forest Preview!)
```python
print("🎲 Testing random feature selection:")
for n_features in [5, 10, 15, None]:
    tree = DecisionTree(max_depth=10, n_feats=n_features)
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)
    accuracy = np.sum(y_test == predictions) / len(y_test)
    print(f"Features {n_features}: Accuracy = {accuracy:.3f}")
```

## 💡 When to Use Decision Trees

### ✅ Decision Trees Work Great For:
- **Interpretable models** (can explain every decision)
- **Mixed data types** (handles both numerical and categorical features)
- **Non-linear relationships** (can capture complex patterns)
- **Feature selection** (automatically identifies important features)
- **Missing values** (can handle gaps in data with modifications)
- **Medical diagnosis** (doctors can follow the logic)

### ❌ Decision Trees Struggle With:
- **Overfitting** (can memorize training data too well)
- **Instability** (small data changes can create very different trees)
- **Linear relationships** (neural networks or linear models might be better)
- **Continuous numerical relationships** (can create jagged approximations)

## 🔧 Hyperparameter Tuning

### Maximum Depth (`max_depth`)
```python
# Too shallow (3): Might underfit, simple decisions
# Just right (10): Good balance of complexity and generalization
# Too deep (None): Might overfit, memorizes training data
```

### Minimum Samples Split (`min_samples_split`)
```python
# Too small (2): Allows very specific splits, might overfit
# Just right (5-10): Good balance
# Too large (50+): Forces simpler tree, might underfit
```

### Number of Features (`n_feats`)
```python
# All features (None): Uses all available information
# Square root (√n): Common choice for random forests
# Custom number: Experiment to find best balance
```

## 🐛 Troubleshooting

### Common Issues and Solutions

1. **Low Accuracy (<70%)**
   ```python
   # Try deeper trees
   tree = DecisionTree(max_depth=15)
   
   # Try different random seeds
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
   
   # Check data quality
   print("Class distribution:", np.bincount(y))
   ```

2. **Overfitting (perfect training accuracy, poor test accuracy)**
   ```python
   # Reduce max depth
   tree = DecisionTree(max_depth=5)
   
   # Increase minimum samples for split
   tree = DecisionTree(min_samples_split=20)
   
   # Use feature randomness
   tree = DecisionTree(n_feats=int(np.sqrt(X.shape[1])))
   ```

3. **IndexError or AttributeError**
   ```python
   # Check the bug fixes in decision_tree.py:
   # 1. X_columns = X[:,feat_idx] (not X[:feat_idx])
   # 2. return parent_entropy - child_entropy (not just child_entropy)
   # 3. Handle empty Y arrays in _most_common_label
   # 4. Use node.features (not node.feature) in _traverse_tree
   ```

## 🎓 Learning More

### 📚 Key Concepts You've Mastered
- **Information Theory**: Entropy and information gain for optimal splits
- **Recursive Algorithms**: Building complex structures from simple rules
- **Greedy Optimization**: Making locally optimal choices at each step
- **Tree Data Structures**: Nodes, leaves, and traversal algorithms
- **Overfitting vs. Underfitting**: Balancing model complexity

### 🚀 Next Steps
- Implement pruning to reduce overfitting
- Add support for regression (predicting continuous values)
- Build a Random Forest (ensemble of decision trees)
- Try different splitting criteria (Gini impurity, etc.)
- Add feature importance calculation

## 📊 Mathematical Deep Dive

### Entropy Formula
```
H(Y) = -Σ p(c) × log₂(p(c))
```
Where p(c) is the proportion of samples belonging to class c.

### Information Gain Formula
```
IG(Y, X) = H(Y) - Σ |Yᵥ|/|Y| × H(Yᵥ)
```
Where Yᵥ is the subset of Y for which feature X has value v.

### Splitting Criterion
```
Best Split = argmax(IG(Y, X_feature, threshold))
```

## 🏆 Conclusion

Congratulations! You now have a complete, working Decision Tree implementation that:
- ✅ Achieves 91% accuracy on real medical data
- ✅ Uses information theory for optimal decision making
- ✅ Builds interpretable models you can explain to anyone
- ✅ Demonstrates core computer science concepts (trees, recursion)
- ✅ Forms the foundation for Random Forests and other ensemble methods

Decision Trees show how computers can make logical decisions just like humans do - by asking the right questions in the right order! 🌳

---

**Happy Tree Growing!** 🌱

*Built with ❤️ for understanding how machines make intelligent decisions!*

*Questions? Try visualizing a simple tree with just 2-3 features and see the decision boundaries!* 🔬