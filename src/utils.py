import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

class FeatureCurve:
    def __init__(self, estimator, X_train, y_train, scoring=balanced_accuracy_score, cv=8, features_order=None):
        self.estimator = estimator
        self.scoring = scoring
        self.cv = StratifiedKFold(cv)
        self.feature_names = X_train.columns

        if features_order is None:
            feature_importance_estimator = GradientBoostingClassifier()
            feature_importance_estimator.fit(X_train, y_train)
            self.features_order_ = X_train.columns[np.argsort(feature_importance_estimator.feature_importances_)[::-1]]

        else:
            self.features_order_ = features_order
        self.train_scores_ = []
        self.test_scores_ = []

    def fit(self, X_train, y_train):
        for fsize in range(1, len(self.features_order_) + 1):
            features = self.features_order_[:fsize]
            iteration_train = []
            iteration_test = []
            for i, (train_index, test_index) in enumerate(self.cv.split(X_train, y_train)):
                X = X_train.iloc[train_index][features]
                y = y_train.iloc[train_index]
                self.estimator.fit(X, y)
                iteration_train.append(self.scoring(self.estimator.predict(X), y))
                X = X_train.iloc[test_index][features]
                y = y_train.iloc[test_index]
                iteration_test.append(self.scoring(self.estimator.predict(X), y))
            self.train_scores_.append(iteration_train)
            self.test_scores_.append(iteration_test)

        self.train_scores_mean_ = np.mean(self.train_scores_, axis=1)
        self.train_scores_std_ = np.std(self.train_scores_, axis=1)
        self.test_scores_mean_ = np.mean(self.test_scores_, axis=1)
        self.test_scores_std_ = np.std(self.test_scores_, axis=1)
        self.draw()
        return self

    def draw(self):
        fig, ax = plt.subplots()
        labels = ("Training Score", "Cross Validation Score")
        curves = (
            (self.train_scores_mean_, self.train_scores_std_),
            (self.test_scores_mean_, self.test_scores_std_),
        )
        colors = ['blue', 'green']

        # Plot the fill betweens first so they are behind the curves.
        for idx, (mean, std) in enumerate(curves):
            # Plot one standard deviation above and below the mean
            ax.fill_between(
                self.features_order_, mean - std, mean + std, alpha=0.25, color=colors[idx]
            )

        # Plot the mean curves so they are in front of the variance fill
        for idx, (mean, _) in enumerate(curves):
            ax.plot(
                self.features_order_, mean, "o-", color=colors[idx], label=labels[idx]
            )
        ax.tick_params(axis='x', labelrotation=80)
        ax.legend()
        ax.set_title('Feature Curve')
        ax.set_ylabel('Score')
        ax.set_xlabel('Used Features')

        return ax

class FeatureLearningCurve:
    def __init__(self, estimator, X_train, y_train, scoring=balanced_accuracy_score, cv=8, features_order=None, train_sizes=None):
        self.estimator = estimator
        self.scoring = scoring
        self.cv = StratifiedKFold(cv)
        self.feature_names = X_train.columns

        if features_order is None:
            feature_importance_estimator = GradientBoostingClassifier()
            feature_importance_estimator.fit(X_train, y_train)
            self.features_order_ = X_train.columns[np.argsort(feature_importance_estimator.feature_importances_)[::-1]]
        else:
            self.features_order_ = features_order

        if train_sizes is None:
            self.train_sizes_ = np.linspace(0.1, 0.99, 5)
        else:
            self.train_sizes_ = train_sizes
        self.train_scores_ = []
        self.test_scores_ = []

    def fit(self, X_train, y_train, X_test, y_test):
        for fsize in range(1, len(self.features_order_) + 1):
            features = self.features_order_[:fsize]
            feature_train = []
            feature_test = []
            for size in self.train_sizes_:
                size_train = []
                size_test = []
                X_reduced, _, y_reduced, _ = train_test_split(X_train, y_train, train_size=size, stratify=y_train)
                for i, (train_index, test_index) in enumerate(self.cv.split(X_reduced, y_reduced)):
                    X = X_reduced.iloc[train_index][features]
                    y = y_reduced.iloc[train_index]
                    self.estimator.fit(X, y)
                    size_train.append(self.scoring(self.estimator.predict(X), y))
                    X = X_test[features]
                    y = y_test
                    size_test.append(self.scoring(self.estimator.predict(X), y))
                feature_train.append(size_train)
                feature_test.append(size_test)
            self.train_scores_.append(feature_train)
            self.test_scores_.append(feature_test)

        self.train_scores_mean_ = np.mean(self.train_scores_, axis=-1)
        self.train_scores_std_ = np.std(self.train_scores_, axis=-1)
        self.test_scores_mean_ = np.mean(self.test_scores_, axis=-1)
        self.test_scores_std_ = np.std(self.test_scores_, axis=-1)
        self.draw()
        return self

    def draw(self, nlayers=10):
        fig, ax = plt.subplots()
        contourf = ax.contourf(self.train_sizes_, self.features_order_, self.test_scores_mean_, nlayers)
        cbar = fig.colorbar(contourf)
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel('Cross Validation Score', rotation=270)

        ax.set_title('Feature-Learning Contour Plot')
        ax.set_ylabel('Used Features')
        ax.set_xlabel('Train Size')

        return ax


def feature_importance_dynamics(estimator, X_train, y_train, X_test, y_test):
    sizes = (np.linspace(0.3, 1.0, 10) * len(X_train)).astype(int)
    importances = []
    for sz in sizes:
        X_partial, y_partial = X_train[:sz], y_train[:sz]
        estimator.fit(X_partial, y_partial)
        importances.append(estimator.feature_importances_)
    importances = np.array(importances).T
    diffs = np.abs(importances[:, -1] - importances[:, 0])
    changed = np.argsort(diffs)[-5:]
    for i, col in enumerate(X_train.columns):
        if i in changed:
            plt.plot(sizes, importances[i], label=col)
    plt.legend()
