import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skforecast.recursive import ForecasterRecursive
from sklearn.model_selection import learning_curve as sk_learning_curve, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier

DEFAULT_TRAIN_SIZES = np.linspace(0.1, 1.0, 10)
DEFAULT_COLORS = ['deepskyblue', 'limegreen', 'violet', 'sandybrown']
SECONDARY_COLORS = ['lightskyblue', 'lightgreen', 'plum', 'peachpuff']


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

    predict_trajectory : boolean, optional, default: True
        Whether to predict the trajectory of learning curves
        using time series forecasting model.

    predict_extend_points : int, optional, default: 5
        Number of train_size points to predict. Used only when
        predict_trajectory is True.

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
        predict_trajectory=True,
        predict_extend_points=5,
        cv=None,
        scoring=None,
        shuffle=False,
        random_state=None
    ):
        # Validate the train sizes
        train_sizes = np.asarray(train_sizes)

        # Set the metric parameters to be used later
        self.estimator = estimator
        self.train_sizes = train_sizes
        self.predict_trajectory = predict_trajectory
        self.predict_extend_points = predict_extend_points
        self.cv = cv
        self.scoring = 'balanced_accuracy' if scoring is None else scoring
        self.shuffle = shuffle
        self.random_state = random_state

    def predict_points(self, points, n_points):
        fr = ForecasterRecursive(
            regressor=LinearRegression(),
            lags=4
        )
        fr.fit(pd.Series(points))
        return fr.predict(steps=n_points).to_numpy()

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
            key: getattr(self, key)
            for key in (
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

        if self.predict_trajectory:
            self.extend_train_sizes_ = self.predict_points(self.train_sizes_, self.predict_extend_points)
            self.extend_train_sizes_ = np.concatenate(([self.train_sizes_[-1]], self.extend_train_sizes_))

            predict_colors = SECONDARY_COLORS[:2]

            extended_curves = []
            for curve in curves:
                extended_curve_stats = []
                for stats in curve:
                    extended_stats = self.predict_points(stats, self.predict_extend_points)
                    extended_curve_stats.append(
                        np.concatenate(([stats[-1]], extended_stats))
                    )
                extended_curves.append(extended_curve_stats)


            # Plot the fill betweens first so they are behind the curves.
            for idx, (mean, std) in enumerate(extended_curves):
                # Plot one standard deviation above and below the mean
                self.ax.fill_between(
                    self.extend_train_sizes_, mean - std, mean + std, alpha=0.25, color=predict_colors[idx]
                )

            for idx, (mean, _) in enumerate(extended_curves):
                self.ax.plot(
                    self.extend_train_sizes_, mean, "o--",
                    color=predict_colors[idx], label='Expected '+labels[idx]
                )

        self.finalize()

        return self.ax

    def finalize(self, **kwargs):
        """
        Add the title, legend, and other visual final touches to the plot.
        """
        # Set the title of the figure
        self.ax.set_title("Learning Curve")

        # Add the legend
        self.ax.legend(frameon=True, loc="best")

        # Set the axis labels
        self.ax.set_xlabel("Training Instances")
        self.ax.set_ylabel("Score")


class FeatureCurve:
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
        features_order=None,
        cv=None,
        scoring=None,
        random_state=None,
        **kwargs
    ):
        self.estimator = estimator
        self.scoring = 'balanced_accuracy' if scoring is None else scoring
        self.cv = cv
        self.random_state = random_state
        self.features_order = features_order


    def fit(self, X, y):
        # Get features_order using gbc estimator
        if self.features_order is None:
            feature_importance_estimator = GradientBoostingClassifier()
            feature_importance_estimator.fit(X, y)
            self.features_order = X.columns[np.argsort(feature_importance_estimator.feature_importances_)[::-1]]

        self.train_scores_ = []
        self.test_scores_ = []
        for fsize in range(1, len(self.features_order) + 1):
            features = self.features_order[:fsize]
            scores = cross_validate(self.estimator, X[features], y, scoring=self.scoring, return_train_score=True)
            self.train_scores_.append(scores['train_score'])
            self.test_scores_.append(scores['test_score'])

        self.train_scores_mean_ = np.mean(self.train_scores_, axis=1)
        self.train_scores_std_ = np.std(self.train_scores_, axis=1)
        self.test_scores_mean_ = np.mean(self.test_scores_, axis=1)
        self.test_scores_std_ = np.std(self.test_scores_, axis=1)
        self.draw()
        return self

    def draw(self):
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
                self.features_order, mean - std, mean + std, alpha=0.25, color=colors[idx]
            )

        # Plot the mean curves so they are in front of the variance fill
        for idx, (mean, _) in enumerate(curves):
            self.ax.plot(
                self.features_order, mean, "o-", color=colors[idx], label=labels[idx]
            )
        self.ax.tick_params(axis='x', labelrotation=90)
        self.finalize()

        return self.ax

    def finalize(self, **kwargs):
        """
        Add the title, legend, and other visual final touches to the plot.
        """
        # Set the title of the figure
        self.ax.set_title("Feature Curve")

        # Add the legend
        self.ax.legend(frameon=True, loc="best")

        # Set the axis labels
        self.ax.set_xlabel("Used Features")
        self.ax.set_ylabel("Score")


class FeatureLearningPlot:
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
        cv=None,
        scoring=None,
        features_order=None,
        train_sizes=DEFAULT_TRAIN_SIZES,
        random_state=None,
        shuffle=False,
        **kwargs
    ):
        self.estimator = estimator
        self.scoring = 'balanced_accuracy' if scoring is None else scoring
        self.cv = cv
        self.features_order = features_order
        self.train_sizes = train_sizes
        self.random_state = random_state
        self.shuffle = shuffle

    def fit(self, X, y):
        # Get features_order using gbc estimator
        if self.features_order is None:
            feature_importance_estimator = GradientBoostingClassifier()
            feature_importance_estimator.fit(X, y)
            self.features_order = X.columns[np.argsort(feature_importance_estimator.feature_importances_)[::-1]]

        sklc_kwargs = {
            key: getattr(self, key)
            for key in (
                "train_sizes",
                "cv",
                "scoring",
                "shuffle",
                "random_state",
            )
        }

        self.train_scores_ = []
        self.test_scores_ = []
        for fsize in range(1, len(self.features_order) + 1):
            features = self.features_order[:fsize]

            curve = sk_learning_curve(self.estimator, X[features], y, **sklc_kwargs)
            self.train_sizes, feature_train_scores, feature_test_scores = curve

            self.train_scores_.append(feature_train_scores)
            self.test_scores_.append(feature_test_scores)

        self.train_scores_mean_ = np.mean(self.train_scores_, axis=-1)
        self.train_scores_std_ = np.std(self.train_scores_, axis=-1)
        self.test_scores_mean_ = np.mean(self.test_scores_, axis=-1)
        self.test_scores_std_ = np.std(self.test_scores_, axis=-1)
        self.draw()
        return self

    def draw(self, nlayers=10):
        self.fig, self.ax = plt.subplots()

        contourf = self.ax.contourf(self.train_sizes, self.features_order, self.test_scores_mean_, nlayers)
        cbar = self.fig.colorbar(contourf)
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel('Cross Validation Score', rotation=270)
        self.finalize()

        return self.ax

    def finalize(self, **kwargs):
        """
        Add the title, legend, and other visual final touches to the plot.
        """
        # Set the title of the figure
        self.ax.set_title('Feature-Learning Contour Plot')

        # Set the axis labels
        self.ax.set_ylabel('Used Features')
        self.ax.set_xlabel('Train Size')
