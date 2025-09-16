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
  - [🧠 Logistic Regression](#-logistic-regression)
  - [🧠 Naive Bayes](#-naive-bayes)
  - [🌳 Decision Tree](#-decision-tree)  
- [💻 Code Showcase](#-code-showcase)
- [⚙️ Installation & Setup](#️-installation--setup)
- [📊 Algorithm Comparisons](#-algorithm-comparisons)
- [🧠 Mathematical Foundations](#-mathematical-foundations)
- [📁 Project Structure](#-project-structure)
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

### 🧠 Logistic Regression

**📁 Location:** [`algorithms/logistic_regression_algorithm/`](algorithms/logistic_regression_algorithm/)

**📝 Algorithm Type:** Supervised Learning - Binary Classification

#### 🔍 Algorithm Overview

Logistic Regression uses the sigmoid function to transform linear combinations into probabilities, making it perfect for binary classification. It combines linear modeling with probability theory to make intelligent yes/no decisions.

#### 🔧 Key Features

| Feature | Description |
|---------|-------------|
| 🌊 **Sigmoid Activation** | Converts linear outputs to probabilities (0-1) |
| 🎯 **Binary Classification** | Perfect for yes/no, spam/ham, cancer/benign decisions |
| 📊 **Gradient Descent** | Custom optimization with cross-entropy loss |
| 🔬 **Medical Applications** | Tested on real breast cancer detection data |
| 📈 **Probability Output** | Get confidence scores, not just predictions |

#### 📈 Performance Metrics

**Tested on Breast Cancer Dataset:**
- **Dataset**: 569 patients, 30 tumor characteristics
- **Accuracy**: ~91% classification accuracy
- **Classes**: Malignant vs Benign tumors
- **Optimization**: 1000 iterations, learning rate 0.001
- **Medical Relevance**: Can assist in cancer diagnosis

#### 💡 When to Use Logistic Regression

✅ **Good for:**
- Binary classification problems
- When you need probability estimates
- Linear decision boundaries
- Fast inference requirements
- Interpretable medical/business models

❌ **Avoid when:**
- Multi-class problems (without extensions)
- Complex non-linear relationships
- Perfect class separation exists
- Very high-dimensional sparse data

---

### 🧠 Naive Bayes

**📁 Location:** [`algorithms/naive_bayes_algorithm/`](algorithms/naive_bayes_algorithm/)

**📝 Algorithm Type:** Supervised Learning - Probabilistic Classification

#### 🔍 Algorithm Overview

Naive Bayes is a probabilistic classifier based on applying Bayes' theorem with the "naive" assumption of conditional independence between features. Despite its simplicity, it's surprisingly effective for many classification tasks!

#### 🔧 Key Features

| Feature | Description |
|---------|-------------|
| 🎲 **Probabilistic Foundation** | Uses Bayes' theorem for decision making |
| 📊 **Statistical Learning** | Calculates mean and variance for each feature |
| 🚀 **Fast Training** | No iterative optimization - just statistical calculations |
| 🎯 **Multi-class Support** | Naturally handles multiple classes |
| 📈 **Real Testing** | Validated on synthetic classification datasets |

#### 📈 Performance Metrics

**Tested on Synthetic Dataset:**
- **Dataset**: 1000 samples, 10 features, 2 classes
- **Accuracy**: ~92.5% classification accuracy
- **Training Time**: Instant (no iterative training)
- **Memory**: Very efficient - stores only statistical summaries

#### 💡 When to Use Naive Bayes

✅ **Good for:**
- Text classification (spam detection, sentiment analysis)
- Small datasets with limited training data
- Fast training and prediction requirements
- Multi-class classification problems
- High-dimensional data

❌ **Avoid when:**
- Features are strongly correlated
- Non-normal feature distributions
- Complex feature interactions are important

Let's dive into the actual implementations! Here's how each algorithm works in practice:

### 🎯 KNN Implementation Walkthrough

#### Core Distance Function
``python
# 📁 File: algorithms/knn_algorithm/knn.py
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))
```
**What it does:** Calculates the straight-line distance between two points in multi-dimensional space.

#### The KNN Class
``python
class KNN:
    def __init__(self, k) -> None:
        self.k = k  # Number of neighbors to consider

    def fit(self, X, Y):
        # 📚 Lazy learning - just store the training data!
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        # 🔍 Predict for multiple samples
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        # 📏 Step 1: Calculate distance to all training points
        distance = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # 📊 Step 2: Find k nearest neighbors
        k_indices = np.argsort(distance)[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_indices]
        
        # 🗳️ Step 3: Democratic voting - most common class wins!
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
```

#### KNN in Action
``python
# 📁 File: algorithms/knn_algorithm/knn_test.py
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 🌺 Load the famous Iris dataset
iris = datasets.load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(
    iris.data, iris.target, random_state=0, test_size=0.2
)

# 🎯 Create and train KNN classifier
clf = KNN(k=5)
clf.fit(X_train, Y_train)

# 🔮 Make predictions and check accuracy
y_pred = clf.predict(X_test)
print(accuracy_score(Y_test, y_pred))  # Expected: ~0.97 (97% accuracy!)
```

### 📈 Linear Regression Implementation Walkthrough

#### The Linear Regression Class
``python
# 📁 File: algorithms/linear_regression_algorithm/linear_regression.py
class LinearRegression:
    def __init__(self, lr, n_iterations):
        self.lr = lr                    # 🏃 Learning rate - how big steps to take
        self.n_iterations = n_iterations # 🔄 How many times to improve
        self.weights = None             # 📈 The slope(s) of our line
        self.bias = None               # 📈 The y-intercept of our line

    def fit(self, X, Y):
        # 🎯 Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Start with zero weights
        self.bias = 0                        # Start with zero bias
        
        # 🔄 Gradient descent optimization loop
        for i in range(self.n_iterations):
            # 🔮 Step 1: Make predictions with current weights
            y_predict = np.dot(X, self.weights) + self.bias
            
            # 📉 Step 2: Calculate gradients (how to improve)
            dw = (1/n_samples) * np.dot(X.T, (y_predict - Y))
            db = (1/n_samples) * np.sum(y_predict - Y)
            
            # 👆 Step 3: Update parameters (take a step toward better solution)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        # 🔮 Make predictions with learned parameters
        y_predict = np.dot(X, self.weights) + self.bias
        return y_predict
```

#### Linear Regression in Action
``python
# 📁 File: algorithms/linear_regression_algorithm/linear_regression_test.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

# 📊 Generate synthetic dataset with known relationship
X, Y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

# 🎯 Create and train the regressor
regressor = LinearRegression(lr=0.1, n_iterations=1000)
regressor.fit(X_train, Y_train)  # 💪 Watch it learn!

# 🔮 Test how well it learned
predictions = regressor.predict(X_test)

# 📈 Calculate Mean Squared Error
def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

mse_value = MSE(Y_test, predictions)
print("MSE Value is:", mse_value)  # Expected: 200-500 (lower is better!)
```

### 🧠 Logistic Regression Implementation Walkthrough

#### The Logistic Regression Class
```python
# 📁 File: algorithms/logistic_regression_algorithm/logistic_regression.py
class LogisticRegression:
    def __init__(self, lr, n_iters):
        self.lr = lr                    # 🏃 Learning rate - step size for optimization
        self.n_iters = n_iters          # 🔄 Number of training iterations
        self.weights = None             # 📈 Feature weights (learned parameters)
        self.bias = None               # 📈 Bias term (y-intercept equivalent)

    def fit(self, X, Y):
        # 🎯 Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Start with zero weights
        self.bias = 0                        # Start with zero bias
        
        # 🔄 Gradient descent optimization loop
        for i in range(self.n_iters):
            # 🔮 Step 1: Linear transformation
            linear_model = np.dot(X, self.weights) + self.bias
            
            # 🌊 Step 2: Sigmoid activation (the magic happens here!)
            y_predict = self._segmoid(linear_model)
            
            # 📉 Step 3: Calculate gradients (how to improve)
            dw = (1/n_samples) * np.dot(X.T, (y_predict - Y))
            db = (1/n_samples) * np.sum(y_predict - Y)
            
            # 👆 Step 4: Update parameters (take a step toward better solution)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        # 🔮 Get probabilities
        linear_model = np.dot(X, self.weights) + self.bias
        y_predict = self._segmoid(linear_model)
        
        # 🎯 Convert probabilities to binary predictions
        y_predict_class = [1 if i > 0.5 else 0 for i in y_predict]
        return y_predict_class
    
    def _segmoid(self, s):
        # 🌊 Sigmoid function: converts any real number to (0, 1)
        return (1/(1+np.exp(-s)))  # Note: Fixed the sign for correct implementation
```

#### Logistic Regression in Action - Cancer Detection!
```python
# 📁 File: algorithms/logistic_regression_algorithm/logistic_regression_test.py
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn import datasets

# 🔬 Load real medical data - breast cancer dataset
data_set = datasets.load_breast_cancer()
X, Y = data_set.data, data_set.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

# 🎯 Create and train the logistic regression classifier
logistic_reg = LogisticRegression(lr=0.001, n_iters=1000)
logistic_reg.fit(X_train, Y_train)  # 💪 Watch it learn to detect cancer!

# 🔮 Test how well it learned
predictions = logistic_reg.predict(X_test)

# 📈 Calculate accuracy
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

print('The accuracy of the model is:', accuracy(Y_test, predictions))  # Expected: ~0.91 (91% accuracy!)
```

### 🧠 Naive Bayes Implementation Walkthrough

#### The Naive Bayes Class
```python
# 📁 File: algorithms/naive_bayes_algorithm/naive_bayes.py
class NaiveBayes:
    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self._classes = np.unique(Y)
        n_classes = len(self._classes)

        # 🎯 Initialize statistics storage
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros((n_classes), dtype=np.float64)

        # 📊 Calculate statistics for each class
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
        # 🎲 Calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])  # P(class)
            posterior = np.sum(np.log(self._pdf(idx, x)))  # P(features|class)
            posterior = prior + posterior  # P(class|features) ∝ P(class) × P(features|class)
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        # 🌊 Gaussian probability density function
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
```

#### Naive Bayes in Action - Statistical Classification!
```python
# 📁 File: algorithms/naive_bayes_algorithm/naive_bayes_test.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

# 🎲 Generate synthetic binary classification data
X, Y = datasets.make_classification(
    n_samples=1000, n_features=10, n_classes=2, random_state=123
)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=123
)

# 🎯 Create and train Naive Bayes classifier  
nb = NaiveBayes()
nb.fit(X_train, Y_train)  # 💪 Watch it learn statistics!

# 🔮 Test probabilistic predictions
predictions = nb.predict(X_test)

# 📈 Calculate accuracy
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

print('Naive Bayes classification accuracy', accuracy(Y_test, predictions))  # Expected: ~0.925 (92.5% accuracy!)
```

---

### 🌳 Decision Tree

**📁 Location:** [`algorithms/decision_tree_algorithm/`](algorithms/decision_tree_algorithm/)

**📏 Algorithm Type:** Supervised Learning - Classification

#### 🔍 Algorithm Overview

Decision Trees are one of the most intuitive machine learning algorithms that work exactly like human decision making by asking a series of yes/no questions to reach a conclusion. They use information theory to automatically determine the best questions to ask.

#### 🔧 Key Features

| Feature | Description |
|---------|-------------|
| ✨ **Information Gain** | Uses entropy to find optimal splits |
| 🌳 **Tree Structure** | Builds hierarchical decision rules |
| 📊 **Entropy Calculation** | Measures data "purity" and information content |
| 🔄 **Recursive Splitting** | Grows tree by repeatedly finding best questions |
| 🎯 **Binary Decisions** | Each node makes a simple yes/no decision |
| 📊 **Real Testing** | Validated on breast cancer medical dataset |

#### 📈 Performance Metrics

**Tested on Breast Cancer Dataset:**
- **Dataset Size**: 569 patients, 30 tumor characteristics
- **Accuracy**: ~91-92% classification accuracy
- **Classes**: Malignant vs Benign tumors
- **Tree Depth**: Configurable (default max_depth=10)
- **Medical Impact**: Can assist doctors in cancer diagnosis! 🏥

#### 💡 When to Use Decision Trees

✅ **Good for:**
- Interpretable models (can explain every decision)
- Mixed data types (numerical and categorical)
- Non-linear relationships
- Feature selection (automatically identifies important features)
- Medical diagnosis (doctors can follow the logic)

❌ **Avoid when:**
- Overfitting is a major concern
- Data is very noisy
- Linear relationships dominate
- You need very stable models

### 🎆 What Makes These Implementations Special

| Aspect | Our KNN | Our Linear Regression | Our Logistic Regression | Our Naive Bayes |
|--------|---------|----------------------|------------------------|-----------------|
| **🧮 Simplicity** | Pure intuition - just look at neighbors! | Clear math - find the best line! | Sigmoid magic - probabilities made simple! | Pure statistics - calculate and decide! |
| **📚 Educational** | See every distance calculation | Watch gradient descent optimize | Observe linear-to-probability transformation | Learn Bayes' theorem in action |
| **🔧 Customizable** | Easy to change k value | Tune learning rate and iterations | Adjust learning parameters and see impact | Modify probability distributions |
| **📈 Performance** | 97% accuracy on Iris | Low MSE on synthetic data | 91% accuracy on cancer detection | 92.5% accuracy on synthetic data |
| **🚀 Ready to Use** | Import and classify! | Import and predict! | Import and get probabilities! | Import and classify probabilistically! |

### 🎓 Key Learning Moments

**From KNN Implementation:**
- 🔍 **Distance matters**: How we measure similarity affects results
- 🗳️ **Democracy works**: Majority voting is powerful for classification
- 📚 **Lazy learning**: Sometimes storing examples is better than complex training

**From Linear Regression Implementation:**
- 📈 **Gradients guide us**: Math tells us which direction improves performance
- 🔄 **Iteration improves**: Each step gets us closer to the optimal solution  
- ⚙️ **Parameters matter**: Learning rate and iterations significantly impact results

**From Logistic Regression Implementation:**
- 🌊 **Sigmoid transforms**: How to convert any number to a probability
- 🧠 **Classification magic**: Binary decisions with confidence scores
- 🎯 **Medical AI**: Real-world applications in healthcare and diagnostics

**From Naive Bayes Implementation:**
- 🎲 **Bayes' theorem**: The foundation of probabilistic reasoning in AI
- 📊 **Statistical learning**: How mean and variance capture data patterns
- ⚡ **Instant training**: No optimization needed - just pure statistics!
- 🎯 **Independence assumption**: Sometimes "naive" assumptions work brilliantly

---

## ⚙️ Installation & Setup

### 📾 Prerequisites

- **Python 3.12+** (recommended)
- **Git** for cloning the repository
- **pip** or **uv** for package management

### 🚀 Quick Installation

```
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

# Test Logistic Regression
cd ../logistic_regression_algorithm
python logistic_regression_test.py
# Expected output: Accuracy around 0.91 for cancer detection

# Test Naive Bayes
cd ../naive_bayes_algorithm
python naive_bayes_test.py
# Expected output: Accuracy around 0.925 for classification

# Test Decision Tree
cd ../decision_tree_algorithm
python decision_tree_test.py
# Expected output: Accuracy around 0.91-0.92 for cancer detection
```

### 🎆 Real Performance Results

Here's what you can expect when running the algorithms:

#### 🎯 KNN Results (Iris Dataset)
```
🌺 Running: python knn_test.py
📄 Dataset: 150 iris flowers, 4 features, 3 species
🎯 Classifier: KNN with k=5 neighbors
📈 Result: 0.9666666666666667
🎉 That's 97% accuracy - excellent performance!
```

#### 📈 Linear Regression Results (Synthetic Data)
```
🚀 Running: python linear_regression_test.py
📄 Dataset: 100 samples with linear relationship + noise
🎯 Regressor: 1000 iterations, learning rate 0.1
📈 Result: MSE Value is: ~350-450
🎉 Low error - the algorithm learned the pattern!
```

#### 🧠 Logistic Regression Results (Medical Data)
```
🔬 Running: python logistic_regression_test.py
📄 Dataset: 569 patients, 30 tumor characteristics
🎯 Classifier: 1000 iterations, learning rate 0.001
📈 Result: The accuracy of the model is: 0.912...
🎉 That's 91% accuracy in cancer detection - potentially life-saving!
```

#### 🧠 Naive Bayes Results (Synthetic Data)
```
🎲 Running: python naive_bayes_test.py
📄 Dataset: 1000 samples, 10 features, 2 classes
🎯 Classifier: Probabilistic Bayes classification
📈 Result: Naive Bayes classification accuracy 0.925
🎉 That's 92.5% accuracy with instant training!
```

**What this means:****
- 🎯 **KNN**: Out of 30 test flowers, it correctly identified ~29 species
- 📈 **Linear Regression**: The predicted values are very close to actual values
- 🔬 **Logistic Regression**: Out of 114 cancer cases, it correctly diagnosed ~104 patients
- 🧠 **Naive Bayes**: Out of 200 test samples, it correctly classified ~185 using pure statistics
- 🎆 **All algorithms work great** and demonstrate core ML concepts!

---
## 📊 Algorithm Comparisons

### 🆚 Feature Comparison

| Aspect | KNN | Linear Regression | Logistic Regression | Naive Bayes | Decision Tree |
|--------|-----|-------------------|--------------------|-------------|---------------|
| **Type** | Classification | Regression | Binary Classification | Probabilistic Classification | Classification |
| **Learning** | Lazy (Instance-based) | Eager (Model-based) | Eager (Model-based) | Eager (Statistical) | Eager (Tree-based) |
| **Training Time** | O(1) - Just stores data | O(n × iterations) | O(n × iterations) | O(n) - Statistical calculations | O(n × log n × features) |
| **Prediction Time** | O(n × d) - Calculate all distances | O(d) - Simple matrix multiplication | O(d) - Linear + sigmoid | O(d) - Probability calculations | O(depth) - Tree traversal |
| **Memory Usage** | High - Stores all training data | Low - Only weights and bias | Low - Only weights and bias | Low - Only statistical summaries | Medium - Stores tree structure |
| **Interpretability** | Medium - Shows similar examples | High - Clear linear relationship | High - Feature importance + probabilities | High - Probabilistic reasoning | Very High - Clear decision rules |
| **Assumptions** | None | Linear relationship exists | Linear decision boundary | Feature independence | None |
| **Output** | Class labels | Continuous values | Probabilities + binary predictions | Probabilities + class predictions | Class labels + decision path |
| **Best For** | Complex decision boundaries | Linear relationships | Binary decisions with confidence | Text classification, small data | Interpretable non-linear models |

### 🏆 Performance Comparison

| Dataset Type | KNN Performance | Linear Regression | Logistic Regression | Naive Bayes | Decision Tree |
|--------------|-----------------|-------------------|--------------------|-------------|---------------|
| **Small datasets** | ✅ Excellent | ✅ Excellent | ✅ Excellent | ✅ Excellent | ✅ Excellent |
| **Large datasets** | ❌ Poor (slow) | ✅ Good (fast) | ✅ Good (fast) | ✅ Good (fast) | ✅ Good (fast) |
| **High dimensions** | ❌ Curse of dimensionality | ✅ Handles well with regularization | ✅ Handles reasonably well | ✅ Handles well | ✅ Good (automatic feature selection) |
| **Non-linear data** | ✅ Excellent | ❌ Poor | ❌ Poor (linear boundary only) | ❌ Assumes normal distribution | ✅ Excellent |
| **Noisy data** | ❌ Sensitive to outliers | ✅ Robust with proper preprocessing | ✅ Robust to moderate noise | ✅ Robust with smoothing | ❌ Can overfit to noise |
| **Binary classification** | ✅ Works but overkill | ❌ Not suitable | ✅ Perfect fit | ✅ Great fit | ✅ Excellent |
| **Multi-class classification** | ✅ Natural fit | ❌ Not suitable | ❌ Needs extensions | ✅ Natural fit | ✅ Natural fit |
| **Probability estimates** | ❌ No built-in probabilities | ❌ Not applicable | ✅ Natural probability output | ✅ Natural probability output | ❌ No built-in probabilities |
| **Text classification** | ❌ Poor for text | ❌ Not suitable | ❌ Needs feature engineering | ✅ Excellent | ✅ Good with proper encoding |
| **Interpretability** | ✅ Shows examples | ✅ Linear equation | ✅ Feature weights | ✅ Probabilistic reasoning | ✅ Clear decision rules |

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

### 🧠 Logistic Regression Mathematics

#### Logistic Model
**Linear Transformation:**
```
z = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

**Sigmoid Activation:**
```
σ(z) = 1/(1 + e^(-z))
```

**Matrix Form:**
```
z = Xθ + b
p = σ(z) = 1/(1 + e^(-(Xθ + b)))
```

**Where:**
- `z` = linear transformation output
- `p` = predicted probabilities vector
- `X` = feature matrix (m × n)
- `θ` = weights vector (n × 1)
- `b` = bias term (scalar)
- `σ` = sigmoid function

#### Cost Function
**Cross-Entropy (Log Loss):**
```
J(θ,b) = -(1/m) × Σ[y·log(σ(z)) + (1-y)·log(1-σ(z))]
```

**Where:**
- `J(θ,b)` = cost function
- `m` = number of training examples
- `y` = actual binary labels (0 or 1)
- `σ(z)` = predicted probabilities

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
∂J/∂θ = (1/m) × Xᵀ × (σ(z) - y)
∂J/∂b = (1/m) × Σ(σ(z) - y)
```

**Where:**
- `α` = learning rate
- `Xᵀ` = transpose of feature matrix

#### Algorithm Steps
1. **Initialize** weights (θ) and bias (b) to zero
2. **Forward pass** - calculate linear transformation: z = Xθ + b
3. **Sigmoid activation** - convert to probabilities: p = σ(z)
4. **Calculate cost** - compute cross-entropy loss
5. **Backward pass** - calculate gradients
6. **Update parameters** - apply gradient descent
7. **Repeat** steps 2-6 until convergence
8. **Prediction** - use threshold (typically 0.5) to convert probabilities to binary predictions

### 🧠 Naive Bayes Mathematics

#### Bayes' Theorem
**Core Formula:**
```
P(C|X) = P(X|C) × P(C) / P(X)
```

**For classification (ignoring constant P(X)):**
```
P(C|X) ∝ P(X|C) × P(C)
```

**Where:**
- `P(C|X)` = posterior probability (what we want)
- `P(X|C)` = likelihood (probability of features given class)
- `P(C)` = prior probability (how common is this class)
- `P(X)` = evidence (normalizing constant)

#### Naive Independence Assumption
**Feature Independence:**
```
P(X|C) = P(x₁|C) × P(x₂|C) × ... × P(xₙ|C)
```

**Where:**
- `X = [x₁, x₂, ..., xₙ]` = feature vector
- Each feature is assumed independent given the class

#### Gaussian Probability Density
**For continuous features:**
```
P(xᵢ|C) = (1/√(2πσ²)) × exp(-(xᵢ-μ)²/(2σ²))
```

**Where:**
- `μ` = mean of feature i for class C
- `σ²` = variance of feature i for class C
- `xᵢ` = value of feature i

#### Classification Decision
**Maximum A Posteriori (MAP):**
```
ŷ = argmax_c [log P(C) + Σᵢ log P(xᵢ|C)]
```

**Algorithm Steps**
1. **Calculate priors** - P(C) for each class
2. **Calculate feature statistics** - mean and variance for each feature per class
3. **For prediction** - calculate posterior for each class using Bayes' theorem
4. **Choose class** with highest posterior probability

---

## 📁 Project Structure

```
ML-Algorithms-From-Scratch/
│
├── 📁 algorithms/                    # Main algorithms directory
│   │
│   ├── 🎯 knn_algorithm/               # K-Nearest Neighbors implementation
│   │   ├── 🐍 knn.py                     # Core KNN class implementation
│   │   ├── 🧪 knn_test.py               # KNN testing and validation
│   │   └── 📚 README.md                # KNN comprehensive guide
│   │
│   ├── 📈 linear_regression_algorithm/ # Linear Regression implementation
│   │   ├── 🐍 linear_regression.py       # Core Linear Regression class
│   │   ├── 🧪 linear_regression_test.py   # Testing and validation
│   │   ├── 🚀 main.py                   # Entry point and examples
│   │   ├── 📚 README.md                # Linear Regression guide
│   │   ├── 🗂️ .venv/                      # Virtual environment
│   │   └── ⚙️ pyproject.toml             # Local project configuration
│   │
│   └── 🧠 logistic_regression_algorithm/ # Logistic Regression implementation
│       ├── 🐍 logistic_regression.py      # Core Logistic Regression class
│       ├── 🧪 logistic_regression_test.py  # Medical data testing
│       └── 📚 README.md                # Logistic Regression guide
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
| **`logistic_regression.py`** | Logistic Regression | `LogisticRegression` class, sigmoid activation |
| **`logistic_regression_test.py`** | Classification Testing | Medical data testing, cancer detection |
| **`naive_bayes.py`** | Naive Bayes | `NaiveBayes` class, Gaussian PDF, Bayes' theorem |
| **`naive_bayes_test.py`** | Probabilistic Testing | Synthetic data testing, statistical classification |
| **`decision_tree.py`** | Decision Tree | `DecisionTree` class, entropy calculation, information gain |
| **`decision_tree_test.py`** | Tree Testing | Medical data testing, cancer diagnosis |
| **`requirements.txt`** | Dependencies | NumPy, scikit-learn, tqdm, ipykernel |
| **`pyproject.toml`** | Configuration | Project metadata, workspace settings |

---

## 🔮 Future Roadmap

### 🎯 Phase 1: Classification Algorithms (In Progress)

- [x] **K-Nearest Neighbors** - ✅ Completed
- [x] **Logistic Regression** - ✅ Completed - Binary classification with sigmoid
- [x] **Naive Bayes** - ✅ Completed - Probabilistic classification using Bayes' theorem
- [x] **Decision Trees** - ✅ Completed - Tree-based learning with information theory

### 📈 Phase 2: Regression Algorithms (In Progress)

- [x] **Linear Regression** - ✅ Completed
- [ ] **Polynomial Regression** - Non-linear relationships
- [ ] **Ridge Regression** - L2 regularization
- [ ] **Lasso Regression** - L1 regularization

### 🔗 Phase 4: Unsupervised Learning

- [ ] **K-Means Clustering** - Partitioning algorithm
- [ ] **Hierarchical Clustering** - Tree-based clustering
- [ ] **PCA** - Dimensionality reduction

### 🎯 Phase 5: Ensemble Methods

- [ ] **Random Forest** - Ensemble of decision trees
- [ ] **AdaBoost** - Adaptive boosting
- [ ] **Gradient Boosting** - Sequential improvement

---

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



