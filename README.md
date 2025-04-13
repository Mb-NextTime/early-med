# earlymed: Machine Learning Visualization Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Add other badges if applicable (e.g., build status, PyPI version) -->

`earlymed` is a Python library designed to assist in machine learning model evaluation and understanding through visualization. It provides tools to plot learning curves, feature performance curves, and combined feature-learning plots, helping users diagnose model behavior, understand data requirements, and perform feature selection.

## Key Features

*   **Learning Curve (`LearningCurve`)**: Visualize model performance (training and cross-validation scores) as the training set size increases. Helps diagnose bias vs. variance issues and overfitting. Includes optional trajectory forecasting using `skforecast`.
*   **Feature Curve (`FeatureCurve`)**: Plot model performance as features are incrementally added based on their importance (determined automatically via Gradient Boosting or provided manually). Helps identify the optimal number of features. Includes optional highlighting of potentially optimal feature sets using the Kneedle algorithm.
*   **Feature-Learning Plot (`FeatureLearningPlot`)**: Generate a contour plot showing model performance across varying training set sizes *and* numbers of features used. Provides a comprehensive overview of how performance scales with both data quantity and feature complexity.

## Installation

**From Source (Recommended for Development):**

```bash
git clone https://github.com/Mb-NextTime/early-med.git # Replace with your repo URL if different
cd early-med
pip install .
```

## Usage Examples

Below are examples demonstrating how to use each visualizer class.

**Setup:**

First, let's import necessary libraries and create some sample data and a model:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Import the visualizers from earlymed
from earlymed import LearningCurve, FeatureCurve, FeatureLearningPlot

# Generate sample classification data
X_raw, y = make_classification(n_samples=1000, n_features=20, n_informative=5,
                           n_redundant=5, n_classes=2, random_state=42)

# Create feature names for Pandas DataFrame (important for FeatureCurve/FeatureLearningPlot)
feature_names = [f'feature_{i}' for i in range(X_raw.shape[1])]
X = pd.DataFrame(X_raw, columns=feature_names)

# Split data (optional, but good practice)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose an estimator
estimator = RandomForestClassifier(n_estimators=50, random_state=42)
```

**1. Learning Curve Example:**

Visualize how the score changes with the amount of training data.

```python
# Instantiate the visualizer
lc_viz = LearningCurve(
    estimator,
    cv=5, # 5-fold cross-validation
    scoring='accuracy', # Use accuracy score
    train_sizes=np.linspace(0.1, 1.0, 8), # Use 8 points from 10% to 100% data
    predict_trajectory=True, # Predict future trend
    predict_extend_points=4, # Predict 4 more points
    random_state=42,
    shuffle=True
)

# Fit the data and draw the plot
lc_viz.fit(X, y)
plt.show() # Display the plot
```

*   **Interpretation:** Look for convergence of training and validation scores. A large gap indicates high variance (overfitting). Low scores for both indicate high bias (underfitting). The predicted trajectory estimates future performance gains with more data.

**2. Feature Curve Example:**

Visualize how the score changes as features are added based on importance.

```python
# Instantiate the visualizer
# Feature importance will be calculated internally using GradientBoostingClassifier
# Requires X to be a Pandas DataFrame with column names
fc_viz = FeatureCurve(
    estimator,
    cv=5,
    scoring='accuracy',
    hint_optimal=True, # Highlight potentially optimal feature counts
    n_hints=3,         # Show up to 3 hints
    random_state=42
    # features_order can be provided manually if needed:
    # features_order=['feature_3', 'feature_0', ...]
)

# Fit the data and draw the plot
fc_viz.fit(X, y) # Use the Pandas DataFrame X
plt.show()
```

*   **Interpretation:** Observe how the cross-validation score changes as more features (ordered by importance) are included. The curve might peak or plateau, suggesting an optimal subset of features. Highlighted regions indicate potential candidates for the best feature count based on score and the "elbow" point detected by Kneedle.

**3. Feature-Learning Plot Example:**

Visualize score changes based on *both* training size and number of features.

```python
# Instantiate the visualizer
# Requires X to be a Pandas DataFrame
flp_viz = FeatureLearningPlot(
    estimator,
    cv=5,
    scoring='accuracy',
    train_sizes=np.linspace(0.2, 1.0, 5), # Fewer points for faster computation
    random_state=42,
    shuffle=True
    # features_order can be provided manually
)

# Fit the data and draw the plot
flp_viz.fit(X, y) # Use the Pandas DataFrame X
plt.show()
```

*   **Interpretation:** The contour plot shows regions of high/low scores. This helps understand the interplay between data quantity and feature complexity. You can identify if adding more data helps more than adding features, or vice-versa, for your specific model and dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
