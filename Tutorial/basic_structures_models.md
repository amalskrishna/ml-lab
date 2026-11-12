# Machine Learning Lab – Model Implementation Templates

These are basic reusable Scikit-learn program structures for common ML algorithms. You can modify the dataset, features, target, and model parameters according to the experiment question.

---

## 1. Common Imports

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
```

## 2. Loading a Dataset

### Built-in dataset

```python
from sklearn.datasets import load_iris

data = load_iris()

X = data.data
y = data.target
```

### CSV dataset

```python
data = pd.read_csv("data.csv")

X = data.drop("target", axis=1)
y = data["target"]
```

### Train-test split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

# CLASSIFICATION ALGORITHMS

## 3. Logistic Regression

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

## 4. K-Nearest Neighbors (KNN)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

## 5. Decision Tree

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

## 6. Support Vector Machine (SVM)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="linear")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

### Other SVM kernels

```python
model = SVC(kernel="linear")
model = SVC(kernel="rbf")
model = SVC(kernel="poly", degree=3)
```

## 7. Artificial Neural Network (ANN / MLP)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation="relu",
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

## 8. Random Forest

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

## 9. AdaBoost

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

## 10. Bagging

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

base_model = DecisionTreeClassifier()

model = BaggingClassifier(
    estimator=base_model,
    n_estimators=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

# REGRESSION ALGORITHMS

## 11. Linear Regression

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

## 12. Polynomial Regression

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()

model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

---

# CLUSTERING ALGORITHMS

## 13. K-Means Clustering

```python
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

data = load_iris()
X = data.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

y_pred = model.fit_predict(X_scaled)

print("Cluster Labels:", y_pred)
print("Inertia:", model.inertia_)
print("Silhouette Score:", silhouette_score(X_scaled, y_pred))
```

---

# 14. Important Evaluation Metrics

## Classification

```python
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

## Regression

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

## Clustering

```python
from sklearn.metrics import silhouette_score

print("Silhouette Score:", silhouette_score(X, y_pred))
print("Inertia:", model.inertia_)
```

---

# 15. Important Points to Remember

| Algorithm | Scaling required? | Main model |
|---|---|---|
| Linear Regression | Usually beneficial | `LinearRegression()` |
| Polynomial Regression | Often beneficial | `LinearRegression()` |
| Logistic Regression | Usually beneficial | `LogisticRegression()` |
| KNN | Yes | `KNeighborsClassifier()` |
| Decision Tree | No | `DecisionTreeClassifier()` |
| SVM | Yes | `SVC()` |
| ANN / MLP | Yes | `MLPClassifier()` |
| Random Forest | No | `RandomForestClassifier()` |
| AdaBoost | Usually no | `AdaBoostClassifier()` |
| Bagging | Depends on base model | `BaggingClassifier()` |
| K-Means | Yes | `KMeans()` |

### Important scaling rule

Always split the dataset before scaling.

Correct:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

- `fit_transform()` learns scaling parameters from training data and applies them.
- `transform()` applies those same parameters to testing data.
- Never fit the scaler separately on the test data.

### Important model differences

- **Classification:** Predicts categories or classes.
- **Regression:** Predicts continuous numerical values.
- **Clustering:** Groups similar data points without requiring target labels.
- **Bagging:** Trains multiple models independently using different samples.
- **Boosting:** Trains models sequentially, focusing on previous errors.
- **Random Forest:** Uses multiple decision trees and combines their predictions.

---

# 16. Quick Experiment Workflow

For most supervised ML experiments, remember this sequence:

1. Import required libraries.
2. Load the dataset.
3. Separate features (`X`) and target (`y`).
4. Split into training and testing data.
5. Scale features if required.
6. Create the ML model.
7. Train using `model.fit()`.
8. Predict using `model.predict()`.
9. Evaluate using suitable metrics.

**Most important commands:**

| Command | Purpose |
|---|---|
| `model.fit(X_train, y_train)` | Train the model |
| `model.predict(X_test)` | Make predictions |
| `model.fit_predict(X)` | Fit and predict clusters |
| `scaler.fit_transform(X_train)` | Learn and apply scaling |
| `scaler.transform(X_test)` | Apply existing scaling |
| `accuracy_score(y_test, y_pred)` | Calculate accuracy |
| `mean_squared_error(y_test, y_pred)` | Calculate MSE |

**Lab tip:** Learn the common workflow first. In most experiments, you only need to change the dataset, model, hyperparameters, and evaluation metric.