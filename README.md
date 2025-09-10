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

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.12 or higher
- Git (for cloning)

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ML-Algorithms-From-Scratch
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Or using uv (recommended):**
   ```bash
   uv sync
   ```

3. **Test KNN Algorithm:**
   ```bash
   cd algorithms/knn_algorithm
   python knn_test.py
   ```
   Expected output: Accuracy score around 0.97

4. **Test Linear Regression:**
   ```bash
   cd algorithms/linear_regression_algorithm
   python linear_regression_test.py
   ```
   Expected output: MSE value indicating model performance

## 📊 Project Structure

```
ML-Algorithms-From-Scratch/
├── algorithms/
│   ├── knn_algorithm/
│   │   ├── knn.py              # KNN implementation
│   │   └── knn_test.py         # KNN testing script
│   └── linear_regression_algorithm/
│       ├── linear_regression.py # Linear regression implementation
│       ├── linear_regression_test.py # Testing script
│       ├── main.py             # Entry point
│       └── pyproject.toml      # Local project config
├── README.md                   # This file!
├── requirements.txt            # Python dependencies
└── pyproject.toml             # Main project configuration
```

## 🔬 Dependencies

- **NumPy (2.3.3)**: Core numerical computations
- **scikit-learn**: Dataset loading and evaluation metrics
- **tqdm (4.66.1)**: Progress bars for long-running operations
- **ipykernel (6.30.1)**: Jupyter notebook support

## 🎓 What You'll Learn

By exploring this project, you'll gain:

- **🧮 Mathematical Foundations**: Understanding the math behind ML algorithms
- **💻 Implementation Skills**: How to translate math into efficient code
- **🔍 Algorithm Internals**: Deep dive into how popular algorithms actually work
- **🛠️ NumPy Mastery**: Advanced array operations and vectorization
- **📊 Model Evaluation**: Proper testing and validation techniques
- **🏗️ Code Architecture**: Clean, modular design principles

## 🧠 Algorithm Deep Dives

### KNN Mathematical Foundation

**Euclidean Distance Formula:**
```
d(x₁, x₂) = √(Σᵢ(x₁ᵢ - x₂ᵢ)²)
```

**Classification Decision:**
```
ŷ = mode(y₁, y₂, ..., yₖ)  # Most frequent class among k neighbors
```

### Linear Regression Mathematical Foundation

**Linear Model:**
```
ŷ = Xw + b
```

**Cost Function (MSE):**
```
J(w,b) = (1/2m) * Σᵢ(ŷᵢ - yᵢ)²
```

**Gradient Descent Updates:**
```
w := w - α * (∂J/∂w)
b := b - α * (∂J/∂b)
```

## 🚀 Running the Examples

### KNN on Iris Dataset
```bash
cd algorithms/knn_algorithm
python knn_test.py
```

**What happens:**
- Loads the classic Iris flower dataset
- Splits into training (80%) and test (20%) sets
- Trains KNN with k=5 neighbors
- Evaluates accuracy on test set
- Should achieve ~97% accuracy!

### Linear Regression on Synthetic Data
```bash
cd algorithms/linear_regression_algorithm
python linear_regression_test.py
```

**What happens:**
- Generates synthetic regression dataset
- Trains linear regression with gradient descent
- Uses learning rate 0.1 for 1000 iterations
- Evaluates using Mean Squared Error
- Lower MSE = better fit!

## 🔧 Customization & Experimentation

### KNN Experiments
```python
# Try different k values
for k in [1, 3, 5, 7, 9]:
    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, knn.predict(X_test))
    print(f"k={k}: accuracy={accuracy:.3f}")
```

### Linear Regression Experiments
```python
# Try different learning rates
for lr in [0.01, 0.1, 0.5]:
    model = LinearRegression(lr=lr, n_iterations=1000)
    model.fit(X_train, y_train)
    mse = MSE(y_test, model.predict(X_test))
    print(f"Learning Rate {lr}: MSE={mse:.3f}")
```

## 🐛 Troubleshooting

**Common Issues:**

1. **Import Errors:**
   - Make sure you're in the correct directory
   - Ensure all dependencies are installed

2. **Low Accuracy in KNN:**
   - Try different k values
   - Check if data needs normalization

3. **High MSE in Linear Regression:**
   - Adjust learning rate (try 0.01 if 0.1 is too high)
   - Increase number of iterations
   - Check for feature scaling needs

## 🔮 Future Roadmap

Exciting algorithms coming next:

- [ ] **Logistic Regression** - Classification with sigmoid function
- [ ] **Decision Trees** - Tree-based learning algorithm
- [ ] **K-Means Clustering** - Unsupervised learning
- [ ] **Naive Bayes** - Probabilistic classifier
- [ ] **Neural Networks** - Multi-layer perceptron from scratch
- [ ] **Support Vector Machine** - Maximum margin classifier
- [ ] **Random Forest** - Ensemble of decision trees

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **🐛 Report bugs** by opening issues
2. **💡 Suggest new algorithms** to implement
3. **📚 Improve documentation** and examples
4. **🧪 Add more test cases** and examples
5. **⚡ Optimize implementations** for better performance

## 📚 Learning Resources

Want to learn more? Check out these resources:

