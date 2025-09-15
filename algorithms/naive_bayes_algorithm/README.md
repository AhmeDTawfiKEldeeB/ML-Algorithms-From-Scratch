# 🧠 Naive Bayes From Scratch

> **Probabilistic classification using Bayes' theorem and the "naive" independence assumption!**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-2.3+-orange.svg)](https://numpy.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-green.svg)](https://scikit-learn.org)

## 🌟 Overview

This directory contains a **complete implementation of Naive Bayes from scratch** using only NumPy! Naive Bayes is a probabilistic classifier based on applying Bayes' theorem with the "naive" assumption of conditional independence between features. Despite its simplicity, it's surprisingly effective for many classification tasks! 🎯

### ✨ Why This Implementation?

- 🎲 **Probabilistic Foundation**: See how Bayes' theorem powers machine learning
- 📊 **Statistical Learning**: Understand mean, variance, and Gaussian distributions in action
- 🎯 **Educational Focus**: Every statistical concept explained clearly
- 🚀 **Fast Performance**: No iterative training - just statistical calculations!
- 📈 **Real Testing**: Validated on synthetic classification datasets

## 📁 What's Inside

```
naive_bayes_algorithm/
├── 🐍 naive_bayes.py      # Main Naive Bayes implementation
├── 🧪 naive_bayes_test.py # Testing script with synthetic data
└── 📚 README.md           # This comprehensive guide!
```

## 🔧 Files Explained

### `naive_bayes.py` - The Probability Engine 🎲

Our main implementation showcases:

```python
class NaiveBayes:
    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self._classes = np.unique(Y)
        n_classes = len(self._classes)

        # Initialize statistics storage
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros((n_classes), dtype=np.float64)

        # Calculate statistics for each class
        for idx, c in enumerate(self._classes):
            X_c = X[Y == c]  # Get samples for this class
            self._mean[idx, :] = X_c.mean(axis=0)    # Feature means
            self._var[idx, :] = X_c.var(axis=0)      # Feature variances
            self._priors[idx] = X_c.shape[0] / float(n_samples)  # Class probability

    def predict(self, X):
        y_predict = [self._predict(x) for x in X]
        return np.array(y_predict)
    
    def _predict(self, x):
        posteriors = []

        # Calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])  # P(class)
            posterior = np.sum(np.log(self._pdf(idx, x)))  # P(features|class)
            posterior = prior + posterior  # P(class|features) ∝ P(class) × P(features|class)
            posteriors.append(posterior)
        
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        # Gaussian probability density function
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
```

### `naive_bayes_test.py` - See It Classify! 🚀

Our test script demonstrates the effectiveness:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Generate synthetic binary classification data
X, Y = datasets.make_classification(
    n_samples=1000, n_features=10, n_classes=2, random_state=123
)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=123
)

# Create and train Naive Bayes classifier
nb = NaiveBayes()
nb.fit(X_train, Y_train)

# Make predictions and evaluate
predictions = nb.predict(X_test)
print("Naive Bayes classification accuracy", accuracy(Y_test, predictions))
```

## 🚀 Quick Start

### 1. Run the Test

```bash
# Navigate to the Naive Bayes directory
cd algorithms/naive_bayes_algorithm

# Run the test script
python naive_bayes_test.py
```

**Expected Output:** `Naive Bayes classification accuracy 0.925` (about 92.5% accuracy! 🎉)


## 🧠 How Naive Bayes Works (The Math Made Simple!)

### The Big Picture 🎯

Naive Bayes uses **Bayes' theorem** to calculate probabilities:

```
P(class|features) = P(features|class) × P(class) / P(features)
```

**Since P(features) is constant for all classes, we can ignore it:**
```
P(class|features) ∝ P(features|class) × P(class)
```

### The "Naive" Assumption 🤔

We assume features are **conditionally independent**:
```
P(features|class) = P(feature₁|class) × P(feature₂|class) × ... × P(featureₙ|class)
```

This is "naive" because features are often correlated, but it works surprisingly well!

### Step-by-Step Learning Process

1. **📊 Calculate Class Statistics (Training)**
   ```python
   # For each class, calculate:
   self._mean[class] = X_class.mean(axis=0)    # Feature means
   self._var[class] = X_class.var(axis=0)      # Feature variances  
   self._priors[class] = n_class / n_total     # Class probability
   ```

2. **🎲 Calculate Probabilities (Prediction)**
   ```python
   # For each class, calculate:
   prior = P(class)                           # How common is this class?
   likelihood = P(features|class)             # How likely are these features?
   posterior = prior × likelihood             # Combined probability
   ```

3. **🏆 Choose Winner**
   ```python
   # Return class with highest posterior probability
   return class_with_max_posterior
   ```

### 📈 Gaussian Probability Density Function

For continuous features, we assume they follow a **normal distribution**:

```python
def _pdf(self, class_idx, x):
    mean = self._mean[class_idx]
    var = self._var[class_idx]
    
    # Gaussian PDF formula
    numerator = np.exp(-((x - mean) ** 2) / (2 * var))
    denominator = np.sqrt(2 * np.pi * var)
    
    return numerator / denominator
```

**What this looks like:**
```
Probability Density
        |    📊
   0.4  |   /  \     Class A: mean=2, var=0.5
   0.3  |  /    \
   0.2  | /      \   📊
   0.1  |/        \ /  \  Class B: mean=5, var=1.0
   0.0  |__________\_____\________________
        0  1  2  3  4  5  6  7  8  Feature Value
```

## 📊 Real-World Performance

### 🎯 Synthetic Dataset Results

Our Naive Bayes achieves **~92.5% accuracy** on synthetic classification data!

**Dataset Details:**
- **Samples:** 1000 data points
- **Features:** 10 (randomly generated)
- **Classes:** 2 (binary classification)
- **Test Split:** 20% (200 samples for testing)
- **Training Time:** Instant! (no iterative optimization)

### 🔍 Why It Works So Well

1. **📊 Statistical Foundation**: Uses proven statistical principles
2. **🚀 No Overfitting**: Simple model with few parameters
3. **📈 Efficient**: No iterative training, just statistical calculations
4. **🎯 Robust**: Works well even with limited data

## 🔬 Understanding the Code

### Why This Implementation Rocks

1. **🎲 Pure Probability**
   ```python
   # You can see exactly how probabilities are calculated:
   prior = np.log(self._priors[idx])              # P(class)
   likelihood = np.sum(np.log(self._pdf(idx, x))) # P(features|class)
   posterior = prior + likelihood                  # P(class|features)
   ```

2. **📊 Statistical Transparency**
   ```python
   # Every statistical concept is clearly implemented:
   self._mean[idx, :] = X_c.mean(axis=0)    # Sample mean
   self._var[idx, :] = X_c.var(axis=0)      # Sample variance
   self._priors[idx] = X_c.shape[0] / n_samples  # Class frequency
   ```

3. **🚀 Computational Efficiency**
   ```python
   # Uses log probabilities to avoid numerical underflow
   posterior = np.sum(np.log(self._pdf(idx, x)))  # Log-sum trick
   ```


## 💡 When to Use Naive Bayes

### ✅ Naive Bayes Works Great For:
- **Text classification** (spam detection, sentiment analysis)
- **Small datasets** (works well with limited data)
- **Fast training** (real-time learning applications)
- **Multi-class problems** (naturally handles multiple classes)
- **Baseline models** (quick first attempt at classification)
- **High-dimensional data** (handles many features well)

### ❌ Naive Bayes Struggles With:
- **Strongly correlated features** (violates independence assumption)
- **Continuous features with non-normal distributions** (assumes Gaussian)
- **Zero probabilities** (if feature value never seen in training)
- **Complex feature interactions** (assumes features don't interact)



## 🐛 Troubleshooting

### Common Issues and Solutions

1. **Low Accuracy (<70%)**
   ```python
   # Check if features are roughly normal distributed
   import matplotlib.pyplot as plt
   plt.hist(X[:, 0])  # Plot first feature
   plt.show()
   
   # Try feature scaling if needed
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **Runtime Warnings (divide by zero)**
   ```python
   # Add small constant to variance to prevent division by zero
   self._var[idx, :] = X_c.var(axis=0) + 1e-9
   ```

3. **Perfect Accuracy (100%)**
   ```python
   # Might indicate data leakage - check your train/test split
   # Or dataset might be too simple
   print("Check if dataset is too easy or has data leakage!")
   ```

## 🎓 Learning More

### 📚 Key Concepts You've Mastered
- **Bayes' Theorem**: The foundation of probabilistic reasoning
- **Conditional Independence**: The "naive" assumption that makes computation tractable
- **Gaussian Distribution**: Modeling continuous features with normal curves
- **Maximum A Posteriori (MAP)**: Choosing the most probable class
- **Log Probabilities**: Numerical stability in probability calculations

### 🚀 Next Steps
- Implement different distributions (multinomial for text data)
- Add Laplace smoothing for better handling of unseen features
- Try Complement Naive Bayes for imbalanced datasets
- Explore feature selection techniques for better performance

## 📊 Mathematical Deep Dive

### Bayes' Theorem
```
P(C|X) = P(X|C) × P(C) / P(X)
```

### Naive Independence Assumption
```
P(X|C) = ∏ᵢ P(xᵢ|C)
```

### Gaussian Probability Density
```
P(xᵢ|C) = (1/√(2πσ²)) × exp(-(xᵢ-μ)²/(2σ²))
```

### Classification Decision
```
ŷ = argmax_c [log P(C) + Σᵢ log P(xᵢ|C)]
```

## 🏆 Conclusion

Congratulations! You now have a complete, working Naive Bayes implementation that:
- ✅ Achieves 92.5% accuracy on synthetic data
- ✅ Demonstrates core probabilistic reasoning concepts
- ✅ Uses efficient statistical calculations (no iterative training!)
- ✅ Handles multi-class classification naturally
- ✅ Provides a foundation for understanding probabilistic ML

Naive Bayes shows how powerful simple statistical principles can be in machine learning. Despite being "naive," it often outperforms much more complex algorithms! 🎯

---