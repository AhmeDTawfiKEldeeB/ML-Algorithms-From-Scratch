# 🤖 ML Algorithms From Scratch

> Building machine learning algorithms from the ground up to understand how they really work!

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-2.3+-orange.svg)](https://numpy.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-green.svg)](https://scikit-learn.org)

## 🎯 Project Overview

Welcome to my educational journey of implementing machine learning algorithms from scratch! This project is all about understanding the mathematics and logic behind popular ML algorithms by building them using only fundamental libraries like NumPy, without relying on high-level frameworks.

**Why build from scratch?** 🤔
- 🧠 **Deep Understanding**: Know exactly how algorithms work under the hood
- 📚 **Learning Experience**: Master the mathematical foundations
- 🔧 **Customization**: Ability to modify and extend algorithms
- 🎯 **Interview Prep**: Common technical interview topic

## 🚀 What's Implemented

### 1. K-Nearest Neighbors (KNN) Classifier 🎯

📁 **Location:** [`algorithms/knn_algorithm/`](algorithms/knn_algorithm/)

A complete KNN classifier implementation that learns by example!

**🔧 Key Features:**
- ✨ **Custom Distance Calculation**: Pure Euclidean distance implementation
- 🎯 **Flexible K Value**: Choose any number of neighbors for classification
- 🗳️ **Majority Voting**: Smart prediction based on neighbor consensus
- 📊 **Real Dataset Testing**: Validated on the famous Iris dataset
- 🚀 **Simple API**: Familiar `fit()` and `predict()` methods

**📈 Performance on Iris Dataset:**
- **Accuracy**: ~97% classification accuracy
- **Dataset**: 150 samples, 4 features, 3 classes
- **Speed**: Instant predictions for small-medium datasets

**💡 Usage Example:**
```python
from algorithms.knn_algorithm.knn import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
knn = KNN(k=5)
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)
```

**🧠 How It Works:**
1. **Store Training Data**: Keep all training samples in memory
2. **Calculate Distances**: Compute Euclidean distance to all training points
3. **Find K Neighbors**: Select the k closest training points
4. **Majority Vote**: Return the most frequent class among neighbors

### 2. Linear Regression 📈

📁 **Location:** [`algorithms/linear_regression_algorithm/`](algorithms/linear_regression_algorithm/)

A gradient descent-based linear regression implementation!

**🔧 Key Features:**
- 📊 **Gradient Descent**: Custom implementation of the optimization algorithm
- ⚙️ **Configurable Learning**: Adjustable learning rate and iterations
- 📐 **Mathematical Foundation**: Pure NumPy implementation of the math
- 🎯 **Regression Tasks**: Perfect for continuous value prediction
- 📈 **MSE Evaluation**: Built-in Mean Squared Error calculation

**💡 Usage Example:**
```python
from algorithms.linear_regression_algorithm.linear_regression import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
regressor = LinearRegression(lr=0.1, n_iterations=1000)
regressor.fit(X_train, y_train)

# Make predictions
predictions = regressor.predict(X_test)
```

**🧠 How It Works:**
1. **Initialize Parameters**: Start with zero weights and bias
2. **Forward Pass**: Calculate predictions using linear equation y = Xw + b
3. **Calculate Loss**: Compute Mean Squared Error
4. **Backward Pass**: Calculate gradients for weights and bias
5. **Update Parameters**: Use gradient descent to minimize loss
6. **Repeat**: Iterate until convergence or max iterations

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

