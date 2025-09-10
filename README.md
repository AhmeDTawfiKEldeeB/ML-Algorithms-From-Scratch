# 🤖 ML Algorithms From Scratch

> **Building machine learning algorithms from the ground up to understand how they really work!**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-2.3+-orange.svg)](https://numpy.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-green.svg)](https://scikit-learn.org)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🚀 Implemented Algorithms](#-implemented-algorithms)
  - [🎯 K-Nearest Neighbors (KNN)](#-k-nearest-neighbors-knn)
  - [📈 Linear Regression](#-linear-regression)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🔧 Usage Examples](#-usage-examples)
- [📊 Algorithm Comparisons](#-algorithm-comparisons)
- [🧠 Mathematical Foundations](#-mathematical-foundations)
- [📁 Project Structure](#-project-structure)
- [🧪 Testing & Validation](#-testing--validation)
- [🎓 Learning Resources](#-learning-resources)
- [🔮 Future Roadmap](#-future-roadmap)

## 🎯 Project Overview

Welcome to the **ML Algorithms From Scratch** project! This is an educational initiative focused on implementing fundamental machine learning algorithms using only basic libraries like NumPy. The goal is to provide clear, understandable implementations that reveal the mathematical foundations behind popular ML techniques.

### 🎪 Why Build From Scratch?

| Benefit | Description |
|---------|-------------|
| 🧠 **Deep Understanding** | Know exactly how algorithms work under the hood |
| 📚 **Mathematical Mastery** | Master the mathematical foundations |
| 🔧 **Customization Power** | Ability to modify and extend algorithms |
| 🎯 **Interview Excellence** | Ace technical interviews with confidence |
| 🔍 **Debug Skills** | Better troubleshooting when things go wrong |

### 🎖️ Project Goals

- ✅ **Educational Focus**: Clear, readable implementations
- ✅ **Mathematical Accuracy**: Correct implementation of algorithms
- ✅ **Practical Examples**: Real dataset testing and validation
- ✅ **Comprehensive Documentation**: Everything you need to understand and use

## 🚀 Implemented Algorithms

### 🎯 K-Nearest Neighbors (KNN)

**📁 Location:** [`algorithms/knn_algorithm/`](algorithms/knn_algorithm/)

**📝 Algorithm Type:** Supervised Learning - Classification

#### 🔍 Algorithm Overview

K-Nearest Neighbors is a simple, intuitive algorithm that classifies data points based on the class of their nearest neighbors. It's a "lazy learning" algorithm that stores all training data and makes decisions at prediction time.

#### 🔧 Key Features

| Feature | Description |
|---------|-------------|
| ✨ **Distance Calculation** | Custom Euclidean distance implementation |
| 🎯 **Flexible K Value** | Adjustable number of neighbors (k=1,3,5,7...) |
| 🗳️ **Majority Voting** | Democratic decision making among neighbors |
| 🚀 **Simple API** | Familiar `fit()` and `predict()` interface |
| 📊 **Real Testing** | Validated on UCI Iris dataset |

#### 📈 Performance Metrics

**Tested on Iris Dataset:**
- **Dataset Size**: 150 samples, 4 features, 3 classes
- **Accuracy**: ~97% classification accuracy
- **Speed**: Instant predictions for small-medium datasets
- **Memory**: Stores all training data (lazy learning)

#### 💡 When to Use KNN

✅ **Good for:**
- Small to medium datasets
- Non-linear decision boundaries
- Multi-class classification
- When interpretability is important

❌ **Avoid when:**
- Very large datasets (slow prediction)
- High-dimensional data (curse of dimensionality)
- Noisy data (sensitive to outliers)

---

### 📈 Linear Regression

**📁 Location:** [`algorithms/linear_regression_algorithm/`](algorithms/linear_regression_algorithm/)

**📝 Algorithm Type:** Supervised Learning - Regression

#### 🔍 Algorithm Overview

Linear Regression finds the best-fitting straight line through data points using gradient descent optimization. It models the relationship between features and continuous target variables.

#### 🔧 Key Features

| Feature | Description |
|---------|-------------|
| 📊 **Gradient Descent** | Custom optimization implementation |
| ⚙️ **Configurable Learning** | Adjustable learning rate and iterations |
| 📐 **Pure Mathematics** | No hidden abstractions, pure NumPy |
| 📈 **MSE Evaluation** | Built-in performance measurement |
| 🎯 **Regression Focus** | Perfect for continuous predictions |

#### 📈 Performance Metrics

**Tested on Synthetic Dataset:**
- **Dataset**: 100 samples, 1 feature + noise
- **Optimization**: 1000 iterations, learning rate 0.1
- **Metric**: Mean Squared Error (MSE)
- **Convergence**: Typically converges within 500 iterations

#### 💡 When to Use Linear Regression

✅ **Good for:**
- Linear relationships between features and target
- Continuous value prediction
- When interpretability is crucial
- Baseline model for comparison

❌ **Avoid when:**
- Complex non-linear relationships
- Categorical target variables
- When overfitting is a major concern

---

## ⚙️ Installation & Setup

### 📾 Prerequisites

- **Python 3.12+** (recommended)
- **Git** for cloning the repository
- **pip** or **uv** for package management

### 🚀 Quick Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd ML-Algorithms-From-Scratch

# 2. Install dependencies (choose one method)

# Method A: Using pip
pip install -r requirements.txt

# Method B: Using uv (recommended - faster)
uv sync
```

### 🧪 Verify Installation

```bash
# Test KNN Algorithm
cd algorithms/knn_algorithm
python knn_test.py
# Expected output: Accuracy score around 0.97

# Test Linear Regression
cd ../linear_regression_algorithm
python linear_regression_test.py
# Expected output: MSE value showing model performance
```

---
## 📊 Algorithm Comparisons

### 🆚 Feature Comparison

| Aspect | KNN | Linear Regression |
|--------|-----|-------------------|
| **Type** | Classification | Regression |
| **Learning** | Lazy (Instance-based) | Eager (Model-based) |
| **Training Time** | O(1) - Just stores data | O(n × iterations) |
| **Prediction Time** | O(n × d) - Calculate all distances | O(d) - Simple matrix multiplication |
| **Memory Usage** | High - Stores all training data | Low - Only weights and bias |
| **Interpretability** | Medium - Shows similar examples | High - Clear linear relationship |
| **Assumptions** | None | Linear relationship exists |
| **Best For** | Complex decision boundaries | Linear relationships |

### 🏆 Performance Comparison

| Dataset Type | KNN Performance | Linear Regression Performance |
|--------------|-----------------|-------------------------------|
| **Small datasets** | ✅ Excellent | ✅ Excellent |
| **Large datasets** | ❌ Poor (slow) | ✅ Good (fast) |
| **High dimensions** | ❌ Curse of dimensionality | ✅ Handles well with regularization |
| **Non-linear data** | ✅ Excellent | ❌ Poor |
| **Noisy data** | ❌ Sensitive to outliers | ✅ Robust with proper preprocessing |

---

## 🧠 Mathematical Foundations

### 🎯 KNN Mathematics

#### Distance Calculation
**Euclidean Distance Formula:**
```
d(x₁, x₂) = √(∑ᵢ₌₁ᵈ (x₁ᵢ - x₂ᵢ)²)
```

**Where:**
- `d(x₁, x₂)` = distance between points x₁ and x₂
- `n` = number of features
- `x₁ᵢ, x₂ᵢ` = values of feature i for points x₁ and x₂

#### Classification Decision
**Majority Voting:**
```
ŷ = mode(y₁, y₂, ..., yₖ)
```

**Where:**
- `ŷ` = predicted class
- `y₁, y₂, ..., yₖ` = classes of k nearest neighbors
- `mode()` = most frequent value

#### Algorithm Steps
1. **Calculate distances** from query point to all training points
2. **Sort distances** in ascending order
3. **Select k nearest** neighbors
4. **Vote** - return most common class among k neighbors

### 📈 Linear Regression Mathematics

#### Linear Model
**Hypothesis Function:**
```
hθ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

**Matrix Form:**
```
ŷ = Xθ + b
```

**Where:**
- `ŷ` = predictions vector
- `X` = feature matrix (m × n)
- `θ` = weights vector (n × 1)
- `b` = bias term (scalar)

#### Cost Function
**Mean Squared Error (MSE):**
```
J(θ,b) = (1/2m) × ∑ᵢ₌₁ᵐ (ŷᵢ - yᵢ)²
```

**Where:**
- `J(θ,b)` = cost function
- `m` = number of training examples
- `ŷᵢ` = predicted value for example i
- `yᵢ` = actual value for example i

#### Gradient Descent
**Weight Update:**
```
θ := θ - α × (∂J/∂θ)
```

**Bias Update:**
```
b := b - α × (∂J/∂b)
```

**Gradients:**
```
∂J/∂θ = (1/m) × Xᵀ × (ŷ - y)
∂J/∂b = (1/m) × ∑(ŷ - y)
```

**Where:**
- `α` = learning rate
- `Xᵀ` = transpose of feature matrix

#### Algorithm Steps
1. **Initialize** weights (θ) and bias (b) to zero
2. **Forward pass** - calculate predictions: ŷ = Xθ + b
3. **Calculate cost** - compute MSE
4. **Backward pass** - calculate gradients
5. **Update parameters** - apply gradient descent
6. **Repeat** steps 2-5 until convergence

---

## 📁 Project Structure

```
ML-Algorithms-From-Scratch/
│
├── 📁 algorithms/                    # Main algorithms directory
│   │
│   ├── 🎯 knn_algorithm/               # K-Nearest Neighbors implementation
│   │   ├── 🐍 knn.py                     # Core KNN class implementation
│   │   └── 🧪 knn_test.py               # KNN testing and validation
│   │
│   └── 📈 linear_regression_algorithm/ # Linear Regression implementation
│       ├── 🐍 linear_regression.py       # Core Linear Regression class
│       ├── 🧪 linear_regression_test.py   # Testing and validation
│       ├── 🚀 main.py                   # Entry point and examples
│       ├── 🗂️ .venv/                      # Virtual environment
│       └── ⚙️ pyproject.toml             # Local project configuration
│
├── 📚 README.md                      # Comprehensive documentation (this file!)
├── 📦 requirements.txt               # Python dependencies
└── ⚙️ pyproject.toml                # Main project configuration
```

### 📂 File Descriptions

| File | Purpose | Key Contents |
|------|---------|-------------|
| **`knn.py`** | KNN Implementation | `KNN` class, `euclidean_distance()` function |
| **`knn_test.py`** | KNN Validation | Iris dataset testing, accuracy measurement |
| **`linear_regression.py`** | Linear Regression | `LinearRegression` class, gradient descent |
| **`linear_regression_test.py`** | Regression Testing | Synthetic data testing, MSE calculation |
| **`requirements.txt`** | Dependencies | NumPy, scikit-learn, tqdm, ipykernel |
| **`pyproject.toml`** | Configuration | Project metadata, workspace settings |

---

## 🔮 Future Roadmap

### 🎯 Phase 1: Classification Algorithms (In Progress)

- [x] **K-Nearest Neighbors** - ✅ Completed
- [ ] **Logistic Regression** - Classification with sigmoid function
- [ ] **Naive Bayes** - Probabilistic classifier
- [ ] **Decision Trees** - Tree-based learning algorithm

### 📈 Phase 2: Regression Algorithms (In Progress)

- [x] **Linear Regression** - ✅ Completed
- [ ] **Polynomial Regression** - Non-linear relationships
- [ ] **Ridge Regression** - L2 regularization
- [ ] **Lasso Regression** - L1 regularization

### 🧠 Phase 3: Neural Networks

- [ ] **Perceptron** - Single neuron classifier
- [ ] **Multi-layer Perceptron** - Deep neural network from scratch
- [ ] **Backpropagation** - Manual gradient calculation

### 🔗 Phase 4: Unsupervised Learning

- [ ] **K-Means Clustering** - Partitioning algorithm
- [ ] **Hierarchical Clustering** - Tree-based clustering
- [ ] **PCA** - Dimensionality reduction

### 🎯 Phase 5: Ensemble Methods

- [ ] **Random Forest** - Ensemble of decision trees
- [ ] **AdaBoost** - Adaptive boosting
- [ ] **Gradient Boosting** - Sequential improvement

---

## 🤝 Contributing Guidelines

### 📝 How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-algorithm`
3. **Follow the code style**: Match existing implementations
4. **Add tests**: Every algorithm needs test cases
5. **Update documentation**: Add to README and docstrings
6. **Submit a pull request**: With detailed description

### 📜 Code Style Guidelines

**Class Structure:**
```python
class AlgorithmName:
    def __init__(self, hyperparameter1, hyperparameter2):
        """Initialize the algorithm with hyperparameters."""
        self.param1 = hyperparameter1
        self.param2 = hyperparameter2
        
    def fit(self, X, y):
        """Train the algorithm on data."""
        # Implementation here
        
    def predict(self, X):
        """Make predictions on new data."""
        # Implementation here
```

---

## 🐛 Troubleshooting

### 🚑 Common Issues

#### 1. **Import Errors**
```
ModuleNotFoundError: No module named 'algorithms'
```
**Solution:**
```bash
# Make sure you're in the correct directory
cd algorithms/knn_algorithm
python knn_test.py
```

#### 2. **Low KNN Accuracy**
```
Accuracy: 0.600 (Expected: ~0.97)
```
**Solutions:**
- Try different k values: `KNN(k=3)` or `KNN(k=7)`
- Scale your features if needed
- Check data quality

#### 3. **High Linear Regression MSE**
```
MSE: 1500.0 (Expected: 200-500)
```
**Solutions:**
- Lower learning rate: `lr=0.01` instead of `lr=0.1`
- Increase iterations: `n_iterations=2000`
- Scale features if needed



