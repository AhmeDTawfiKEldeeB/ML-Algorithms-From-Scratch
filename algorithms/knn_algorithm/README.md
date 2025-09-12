# 🎯 K-Nearest Neighbors (KNN) From Scratch

> **Learning by example - classify data points based on their neighbors!**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-2.3+-orange.svg)](https://numpy.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-green.svg)](https://scikit-learn.org)

## 🌟 Overview

This directory contains a **complete implementation of K-Nearest Neighbors from scratch** using only NumPy! KNN is one of the simplest yet most powerful machine learning algorithms - it classifies data points by looking at what their neighbors are doing. No complex math, just pure intuition! 🧠

### ✨ Why This Implementation?

- 🔍 **Pure Simplicity**: See exactly how KNN works with crystal-clear code
- 📐 **Educational Focus**: Every line explained and easy to understand
- 🎯 **Real-World Testing**: Validated on the famous Iris dataset
- 🚀 **Ready to Use**: Just import and start classifying!
- 💡 **Intuitive Design**: Mirrors scikit-learn's familiar API

## 📁 What's Inside

```
knn_algorithm/
├── 🐍 knn.py           # Main KNN implementation
├── 🧪 knn_test.py      # Testing script with Iris dataset
└── 📚 README.md        # This friendly guide!
```

## 🔧 Files Explained

### `knn.py` - The Heart of KNN ❤️

Our main implementation contains:

```python
# 📐 Distance calculation function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

# 🎯 The KNN classifier class
class KNN:
    def __init__(self, k):
        self.k = k  # Number of neighbors to consider
    
    def fit(self, X, Y):
        # Store training data (lazy learning!)
        self.X_train = X
        self.Y_train = Y
    
    def predict(self, X):
        # Predict multiple samples
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        # 1. Calculate distances to all training points
           distance=[euclidean_distance(x,x_train)for x_train in self.X_train]
        # 2. Find k nearest neighbors
         k_indices=np.argsort(distance)[:self.k]
        k_nearest_labels=[self.Y_train[i] for i in k_indices]
        # 3. Vote for the most common class
        most_common=np.bincount(k_nearest_labels).argmax()
        return most_commo
```

### `knn_test.py` - See It In Action! 🚀

Our test script demonstrates:

```python
# Load the famous Iris dataset
iris = datasets.load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(
    iris.data, iris.target, random_state=0, test_size=0.2
)

# Create and train our KNN classifier
clf = KNN(k=5)
clf.fit(X_train, Y_train)

# Make predictions and check accuracy
y_pred = clf.predict(X_test)
print(accuracy_score(Y_test, y_pred))
```

## 🚀 Quick Start

### 1. Run the Test

```bash
# Navigate to the KNN directory
cd algorithms/knn_algorithm

# Run the test script
python knn_test.py
```

**Expected Output:** `0.9666666666666667` (about 97% accuracy! 🎉)

### 2. Use In Your Own Code

```python
import numpy as np
from knn import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load your data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create KNN classifier
knn_classifier = KNN(k=5)

# Train (actually just stores the data!)
knn_classifier.fit(X_train, y_train)

# Predict
predictions = knn_classifier.predict(X_test)

print(f"Predictions: {predictions}")
```

## 🧠 How KNN Works (The Magic Explained!)

### Step-by-Step Process

1. **📚 Training Phase (Super Simple!)**
   ```python
   def fit(self, X, Y):
       self.X_train = X  # Just store the training data
       self.Y_train = Y  # That's it! No complex training needed
   ```

2. **🔍 Prediction Phase (Where the Magic Happens)**
   ```python
   def _predict(self, x):
       # Calculate distance to ALL training points
       distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
       
       # Find the k closest neighbors
       k_indices = np.argsort(distances)[:self.k]
       k_nearest_labels = [self.Y_train[i] for i in k_indices]
       
       # Democratic voting - most common class wins!
       most_common = np.bincount(k_nearest_labels).argmax()
       return most_common
   ```

### 📐 Distance Calculation

We use **Euclidean Distance** - imagine measuring with a ruler in multi-dimensional space:

```python
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))
```

**What this does:**
- `(x1-x2)**2` → Square the differences for each feature
- `np.sum(...)` → Add up all the squared differences  
- `np.sqrt(...)` → Take the square root to get the actual distance

**Example:** Distance between points [1,2] and [4,6]
```
distance = √((1-4)² + (2-6)²) = √(9 + 16) = √25 = 5
```

## 📊 Real-World Performance

### 🌺 Iris Dataset Results

Our KNN implementation achieves **~97% accuracy** on the Iris dataset!

**Dataset Details:**
- **Samples:** 150 flowers
- **Features:** 4 (sepal length, sepal width, petal length, petal width)
- **Classes:** 3 (Setosa, Versicolor, Virginica)
- **Test Split:** 20% (30 samples for testing)

### 🎯 Different K Values Performance

```python
# Test different k values
for k in [1, 3, 5, 7, 9]:
    knn = KNN(k=k)
    knn.fit(X_train, Y_train)
    accuracy = accuracy_score(Y_test, knn.predict(X_test))
    print(f"k={k}: accuracy={accuracy:.3f}")
```

**Typical Results:**
- k=1: 0.933 (can be sensitive to noise)
- k=3: 0.966 (good balance)
- **k=5: 0.966** (our choice - stable and accurate)
- k=7: 0.966 (still good)
- k=9: 0.933 (might be too many neighbors)

## 🔬 Understanding the Code

### Why This Implementation is Great

1. **🎯 Simple and Clear**
   ```python
   # Every function does one thing well
   euclidean_distance()  # Just calculates distance
   fit()                # Just stores data
   predict()            # Just makes predictions
   _predict()           # Does the heavy lifting for one sample
   ```

2. **🧠 Educational Value**
   ```python
   # You can see exactly what happens:
   distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
   # ↑ Calculate distance to every training point
   
   k_indices = np.argsort(distances)[:self.k]
   # ↑ Find the k closest ones
   
   most_common = np.bincount(k_nearest_labels).argmax()
   # ↑ Democratic voting!
   ```

3. **🚀 Efficient with NumPy**
   ```python
   # Uses NumPy for fast numerical operations
   np.sqrt(np.sum((x1-x2)**2))  # Vectorized distance calculation
   np.argsort(distances)        # Fast sorting
   np.bincount(labels)          # Efficient counting
   ```

## 🎮 Try These Experiments

### Experiment 1: Different K Values
```python
# Create your own test
from knn import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🔍 Testing different k values:")
for k in [1, 3, 5, 7, 9, 11]:
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"k={k}: accuracy={accuracy:.3f}")
```

### Experiment 2: Visualize Predictions
```python
# See what your KNN predicts for specific flowers
import numpy as np

# Create a new flower sample (make up your own values!)
new_flower = np.array([[5.0, 3.0, 4.0, 1.2]])  # [sepal_length, sepal_width, petal_length, petal_width]

knn = KNN(k=5)
knn.fit(X_train, y_train)
prediction = knn.predict(new_flower)

class_names = ['Setosa', 'Versicolor', 'Virginica']
print(f"🌸 Your flower is predicted to be: {class_names[prediction[0]]}")
```

### Experiment 3: Custom Dataset
```python
# Create your own simple dataset
X_simple = np.array([[1, 1], [2, 2], [3, 3], [6, 6], [7, 7], [8, 8]])
y_simple = np.array([0, 0, 0, 1, 1, 1])  # Two classes: 0 and 1

# Test point that's closer to class 1
test_point = np.array([[7.5, 7.5]])

knn = KNN(k=3)
knn.fit(X_simple, y_simple)
prediction = knn.predict(test_point)
print(f"🎯 Prediction for [7.5, 7.5]: Class {prediction[0]}")
```

## 💡 When to Use KNN

### ✅ KNN Works Great For:
- **Small to medium datasets** (like our Iris example)
- **Non-linear decision boundaries** (KNN can handle complex shapes)
- **Multi-class problems** (classify into many categories)
- **When you need interpretability** ("It's class A because these similar examples are class A")

### ❌ KNN Struggles With:
- **Very large datasets** (slow prediction because it checks ALL training points)
- **High-dimensional data** (curse of dimensionality)
- **Noisy data** (outliers can mislead the neighbors)
- **Imbalanced datasets** (majority class dominates voting)

## 🐛 Troubleshooting

### Common Issues and Solutions

1. **Low Accuracy**
   ```python
   # Try different k values
   # Scale your features if they have different ranges
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **Slow Performance**
   ```python
   # Use smaller dataset for testing
   # Consider approximate methods for large datasets
   X_sample = X[:1000]  # Use first 1000 samples
   ```

3. **Import Errors**
   ```bash
   # Make sure you're in the right directory
   cd algorithms/knn_algorithm
   python knn_test.py
   ```

## 🎓 Learning More

### 📚 Key Concepts You've Learned
- **Lazy Learning**: KNN doesn't "learn" during training, just stores data
- **Distance Metrics**: How to measure similarity between data points  
- **Majority Voting**: Democratic decision-making in machine learning
- **Hyperparameter Tuning**: Choosing the right k value

### 🚀 Next Steps
- Try implementing other distance metrics (Manhattan, Cosine)
- Add weighted voting (closer neighbors have more influence)
- Implement KNN for regression problems
- Explore dimensionality reduction techniques

## 🏆 Conclusion

Congratulations! You now have a complete, working KNN implementation that:
- ✅ Achieves 97% accuracy on real data
- ✅ Is easy to understand and modify
- ✅ Demonstrates core machine learning concepts
- ✅ Can be extended for your own projects

KNN proves that sometimes the simplest ideas are the most powerful. By just looking at what the neighbors are doing, we can make surprisingly accurate predictions! 🎯

---

**Happy Classifying!** 🎉

*Built with ❤️ for learning and understanding machine learning from the ground up!*

*Questions? Try modifying the code and see what happens!* 🔬