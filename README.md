# 🤖 ML Algorithms From Scratch

> Building machine learning algorithms from the ground up to understand how they really work!

## 🎯 Project Overview

Welcome to my journey of implementing machine learning algorithms from scratch! This project focuses on building ML algorithms using only fundamental libraries like NumPy, helping to understand the mathematics and logic behind popular algorithms without relying on high-level frameworks.

## 🚀 What's Implemented

### K-Nearest Neighbors (KNN) Classifier

📁 **Location:** `algorithms/knn_algorithm/`

I've implemented a complete KNN classifier that:

- ✨ **Custom Distance Calculation**: Uses Euclidean distance to measure similarity between data points
- 🎯 **Flexible K Value**: Choose any number of neighbors for classification
- 🗳️ **Majority Voting**: Makes predictions based on the most common class among k nearest neighbors
- 📊 **Real Dataset Testing**: Validated on the famous Iris dataset with great results!

#### 🔧 Key Features

- **Simple API**: Easy-to-use `fit()` and `predict()` methods just like scikit-learn
- **Pure Python Implementation**: Built from scratch using only NumPy for numerical operations
- **Educational Focus**: Clear, readable code that shows exactly how KNN works under the hood

#### 📈 Performance

Tested on the Iris dataset (150 samples, 4 features, 3 classes):
- **Accuracy**: Achieves excellent classification accuracy
- **Speed**: Fast predictions suitable for small to medium datasets

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd ML-Algorithms-From-Scratch
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the KNN example**:
   ```bash
   cd algorithms/knn_algorithm
   python knn_test.py
   ```

## 💡 How to Use KNN

```python
from algorithms.knn_algorithm.knn import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load your dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the model
knn = KNN(k=5)  # Use 5 nearest neighbors
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)
```

## 🧠 How KNN Works

My implementation follows these steps:

1. **Store Training Data**: Keep all training samples and labels in memory
2. **Calculate Distances**: For each test point, compute Euclidean distance to all training points
3. **Find Neighbors**: Select the k closest training points
4. **Vote**: Return the most frequent class among the k neighbors

### 📐 Distance Formula

The Euclidean distance between two points is calculated as:

```
distance = √(Σ(x₁ᵢ - x₂ᵢ)²)
```

## 🎓 What I Learned

- **Algorithm Internals**: Deep understanding of how KNN makes classifications
- **NumPy Mastery**: Efficient array operations and mathematical computations
- **Code Structure**: Clean, modular design following OOP principles
- **Testing**: Proper validation using real datasets and metrics

## 🔮 Coming Next

I'm planning to implement more algorithms from scratch:

- [ ] Linear Regression
- [ ] Logistic Regression  
- [ ] Decision Trees
- [ ] Neural Networks
- [ ] K-Means Clustering




*Built with ❤️ for learning and understanding ML algorithms at their core!*