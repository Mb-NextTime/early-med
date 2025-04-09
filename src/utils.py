import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve as sk_learning_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

DEFAULT_TRAIN_SIZES = np.linspace(0.1, 1.0, 10)
DEFAULT_COLORS = ['lightblue', 'lightgreen', 'plum', 'peachpuff']

class LearningCurve:
    """
    Visualizes the learning curve for both test and training data for
    different training set sizes. These curves can act as a proxy to
    demonstrate the implied learning rate with experience (e.g. how much data
    is required to make an adequate model). They also demonstrate if the model
    is more sensitive to error due to bias vs. error due to variance and can
    be used to quickly check if a model is overfitting.

    The visualizer evaluates cross-validated training and test scores for
    different training set sizes. These curves are plotted so that the x-axis
    is the training set size and the y-axis is the score.

    Parameters
    ----------
    estimator :
        An object that implements ``fit`` and ``predict``, can be a
        classifier, regressor, or clusterer so long as there is also a valid
        associated scoring metric.

        Note that the object is cloned for each validation.

    train_sizes : array-like, shape (n_ticks,)
        default: ``np.linspace(0.1,1.0,10)``

        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as
        a fraction of the maximum size of the training set, otherwise it is
        interpreted as absolute sizes of the training sets.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

    scoring : string, callable or None, optional, default: `balanced_accuracy_score`
        A string or scorer callable object / function with signature
        ``scorer(estimator, X, y)``. See scikit-learn model evaluation
        documentation for names of possible metrics.

    shuffle : boolean, optional
        Whether to shuffle training data before taking prefixes of it
        based on``train_sizes``.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` is True.

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers.

    Attributes
    ----------
    train_sizes_ : array, shape = (n_unique_ticks,), dtype int
        Numbers of training examples that has been used to generate the
        learning curve. Note that the number of ticks might be less
        than n_ticks because duplicate entries will be removed.

    train_scores_ : array, shape (n_ticks, n_cv_folds)
        Scores on training sets.

    train_scores_mean_ : array, shape (n_ticks,)
        Mean training data scores for each training split

    train_scores_std_ : array, shape (n_ticks,)
        Standard deviation of training data scores for each training split

    test_scores_ : array, shape (n_ticks, n_cv_folds)
        Scores on test set.

    test_scores_mean_ : array, shape (n_ticks,)
        Mean test data scores for each test split

    test_scores_std_ : array, shape (n_ticks,)
        Standard deviation of test data scores for each test split
    """

    def __init__(
        self,
        estimator,
        train_sizes=DEFAULT_TRAIN_SIZES,
        cv=None,
        scoring=None,
        shuffle=False,
        random_state=None,
        **kwargs
    ):
        # Validate the train sizes
        train_sizes = np.asarray(train_sizes)

        # Set the metric parameters to be used later
        self.estimator = estimator
        self.train_sizes = train_sizes
        self.cv = cv
        self.scoring = balanced_accuracy_score if scoring is None else scoring
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fits the learning curve with the wrapped model to the specified data.
        Draws training and test score curves and saves the scores to the
        estimator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features)
            Target relative to X for classification.

        Returns
        -------
        self : instance
            Returns the instance of the learning curve visualizer for use in
            pipelines and other sequential transformers.
        """
        # arguments to pass to sk_learning_curve
        sklc_kwargs = {
            key: self.get_params()[key]
            for key in (
                "groups",
                "train_sizes",
                "cv",
                "scoring",
                "shuffle",
                "random_state",
            )
        }

        # compute the learning curve and store the scores on the estimator
        curve = sk_learning_curve(self.estimator, X, y, **sklc_kwargs)
        self.train_sizes_, self.train_scores_, self.test_scores_ = curve

        # compute the mean and standard deviation of the training data
        self.train_scores_mean_ = np.mean(self.train_scores_, axis=1)
        self.train_scores_std_ = np.std(self.train_scores_, axis=1)

        # compute the mean and standard deviation of the test data
        self.test_scores_mean_ = np.mean(self.test_scores_, axis=1)
        self.test_scores_std_ = np.std(self.test_scores_, axis=1)

        # draw the curves on the current axes
        self.draw()
        return self

    def draw(self, **kwargs):
        """
        Renders the training and test learning curves.
        """
        self.fig, self.ax = plt.subplots()
        # Specify the curves to draw and their labels
        labels = ("Training Score", "Cross Validation Score")
        curves = (
            (self.train_scores_mean_, self.train_scores_std_),
            (self.test_scores_mean_, self.test_scores_std_),
        )

        # Get the colors for the train and test curves
        colors = DEFAULT_COLORS[:2]

        # Plot the fill betweens first so they are behind the curves.
        for idx, (mean, std) in enumerate(curves):
            # Plot one standard deviation above and below the mean
            self.ax.fill_between(
                self.train_sizes_, mean - std, mean + std, alpha=0.25, color=colors[idx]
            )

        # Plot the mean curves so they are in front of the variance fill
        for idx, (mean, _) in enumerate(curves):
            self.ax.plot(
                self.train_sizes_, mean, "o-", color=colors[idx], label=labels[idx]
            )

        return self.ax

    def finalize(self, **kwargs):
        """
        Add the title, legend, and other visual final touches to the plot.
        """
        # Set the title of the figure
        self.set_title("Learning Curve for {}".format(self.name))

        # Add the legend
        self.ax.legend(frameon=True, loc="best")

        # Set the axis labels
        self.ax.set_xlabel("Training Instances")
        self.ax.set_ylabel("Score")


class FeatureCurve:
    """
    """
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
        ax.tick_params(axis='x', labelrotation=85)
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
