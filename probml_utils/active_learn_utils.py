# sklearn Data
from sklearn.datasets import make_classification

# sklearn Utils
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.model_selection import train_test_split

# sklearn Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

import numpy as np

from typing import Callable, Tuple, Optional, List, Union
import subprocess
import sys

try:
    import modAL
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "modAL"])
    import modAL

from modAL.models import ActiveLearner
from modAL.models import Committee
from modAL.models import CommitteeRegressor
from modAL.utils.data import modALinput


def model_scores(
    model: BaseEstimator,
    problem_type: str,
    X_data: np.ndarray,
    y_data: np.ndarray,
    query_type: str,
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Based on problem type and Active Learning Method return relevant metrics

        Classification : return accuracy score

        Regression with Uncertainty/Random : Return RMSE along with
        standard deviation

        Regression with QBC : Return only RMSE

    Args:
    ----------
    model :
        The fitted model

    problem_type :
        Classification or Regression Problem

    X_data :
        Features of test data

    y_data :
        Labels of test data

    query_type:
        QBC or uncertainty or random

    Returns:
    ----------
    acc/rmse/rmse,y_std:
        Returns Accuracy Score or RMSE or
        RMSE along with standard deviation for each point

    """

    if problem_type == "Classification":
        y_pred = model.predict(X_data)
        acc = accuracy_score(y_data, y_pred)
        return acc
    else:
        # Return Standard deviation for regression problem
        if query_type != "QBC":
            y_pred, y_std = model.predict(X_data, return_std=True)
            rmse = mean_squared_error(y_data, y_pred, squared=False)
            return rmse, y_std
        else:
            y_pred = model.predict(X_data)
            rmse = mean_squared_error(y_data, y_pred, squared=False)
            return rmse


def setup_committe(model_list: List[BaseEstimator], X_train: np.ndarray, y_train: np.ndarray) -> List[ActiveLearner]:
    """
    Create and return a list of active learners based
    on inputs model_list

    Args:
    ----------
    model_list :
        The list of models to be converted into active learners

    Returns:
    ----------
    learner_list:
        List of active learners

    X_train:
        Features of train data

    y_train:
        Labels of train data

    """

    learner_list = []

    for model in model_list:

        # initializing learner
        learner = ActiveLearner(
            estimator=model,
            X_training=X_train.copy(),
            y_training=y_train.copy(),
        )

        learner_list.append(learner)

    return learner_list


def qbc(
    sampling: Callable,
    committe_list: list,
    problem_type: str,
    n_queries: int,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Active Learning Loop for QBC

    Args:
    ----------
    sampling :
        The query stratergy function used for sampling

    committe_list :
        A list of members for the committe

    problem_type :
        Classification or Regression Problem

    n_queries :
        The number of queries

    X_pool:
        Features pool data

    y_pool:
        Labels pool data

    X_train:
        Features train data

    y_train:
        Labels train data

    X_test:
        Features test data

    y_train:
        Labels test data


    Returns:
    ----------
    data_dict :
        Dictionary with log of all the points queried along with
        other info
    """

    # Make a copy of pool data
    X_act_pool = X_pool.copy()
    y_act_pool = y_pool.copy()

    # Dictionary to store useful data for plotting
    data_dict = {
        "score": [],
        "pool": {"X": [X_act_pool], "y": [y_act_pool]},
        "train": {"X": [X_train], "y": [y_train]},
        "query_sample": {"X": [], "y": []},
    }

    # Setup Committe
    learner_list = setup_committe(committe_list, X_train, y_train)

    if problem_type == "Classification":
        committe = Committee(learner_list=learner_list, query_strategy=sampling)
    else:
        committe = CommitteeRegressor(learner_list=learner_list, query_strategy=sampling)

    # Scores on initial train data
    score = model_scores(committe, problem_type, X_test, y_test, "QBC")
    data_dict["score"].append(score)

    # Active Learning loop
    for i in range(n_queries):
        query_idx, _ = committe.query(X_act_pool)

        # Store queried sample
        data_dict["query_sample"]["X"].append(X_act_pool[query_idx])
        data_dict["query_sample"]["y"].append(y_act_pool[query_idx])

        # Teach committe
        committe.teach(X=X_act_pool[query_idx], y=y_act_pool[query_idx])

        # remove queried instance from pool
        X_act_pool = np.delete(X_act_pool, query_idx, axis=0)
        y_act_pool = np.delete(y_act_pool, query_idx, axis=0)

        # Keep a track of pool
        data_dict["pool"]["X"].append(X_act_pool)
        data_dict["pool"]["y"].append(y_act_pool)

        # Log the score/error
        score = model_scores(committe, problem_type, X_test, y_test, "QBC")
        data_dict["score"].append(score)

    return data_dict


def uncertainty_sampling(
    sampling: Callable,
    model: BaseEstimator,
    strat: str,
    n_queries: int,
    problem_type: str,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:

    """
    Active Learning Loop for Random and Uncertainty

    Args:
    ----------
    sampling :
        The query stratergy function used for sampling

    model :
        The machine learning model

    strat :
        Name of query stratergy

    n_queries :
        The number for iteration to run for each sampling technique

    problem_type :
        Classification or Regression Problem

    X_pool:
        Features pool data

    y_pool:
        Labels pool data

    X_train:
        Features train data

    y_train:
        Labels train data

    Returns:
    ----------
    data_dict :
        Dictionary with log of all the points queried along with
        other info
    """

    # Declare Model for active learning
    active_model = modAL.models.ActiveLearner(
        estimator=model, query_strategy=sampling, X_training=X_train, y_training=y_train
    )

    # Make a copy of Pool data
    X_act_pool = X_pool.copy()
    y_act_pool = y_pool.copy()

    # Dictionary to store history of query points and pool, helpful for graph plotting
    data_dict = {
        "score": [],
        "std_dev_pts": [],
        "pool": {"X": [X_act_pool], "y": [y_act_pool]},
        "train": {"X": [X_train], "y": [y_train]},
        "query_sample": {"X": [], "y": []},
    }

    # Intial score on train data
    if problem_type == "Classification":
        score = model_scores(active_model, problem_type, X_test, y_test, "Uncertanity")
        data_dict["score"].append(score)
    else:
        score, y_test_std = model_scores(active_model, problem_type, X_test, y_test, "Uncertanity")
        data_dict["score"].append(score)
        data_dict["std_dev_pts"].append(y_test_std)

    # Active Learning loop
    while n_queries:

        # Query a Point
        query_idx, _ = active_model.query(X_act_pool, n_instances=1)

        # Log queried points
        data_dict["query_sample"]["X"].append(X_act_pool[query_idx])
        data_dict["query_sample"]["y"].append(y_act_pool[query_idx])

        # Teach the model the new queried point
        active_model.teach(X_act_pool[query_idx], y_act_pool[query_idx])

        # Delete point from pool
        X_act_pool = np.delete(X_act_pool, query_idx, axis=0)
        y_act_pool = np.delete(y_act_pool, query_idx, axis=0)

        # Log pool data
        data_dict["pool"]["X"].append(X_act_pool)
        data_dict["pool"]["y"].append(y_act_pool)

        # Log the score/error
        if problem_type == "Classification":
            score = model_scores(active_model, problem_type, X_test, y_test, "Uncertanity")
            data_dict["score"].append(score)
        else:
            score, y_test_std = model_scores(active_model, problem_type, X_test, y_test, "Uncertanity")
            data_dict["score"].append(score)
            data_dict["std_dev_pts"].append(y_test_std)

        n_queries -= 1

    return data_dict


def random_sampling(model: BaseEstimator, X: modALinput, n_instances: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random query stratergy for sampling

    Args:
    ----------
    model :
        Classification or Regression model used

    X :
        The data to query from, this would be pool data

    Returns:
    ----------
    query_idx:
        Index of queried sample

    query_sample:
        The actual queried sample
    """

    np.random.seed(None)
    indices = np.random.choice(range(X.shape[0]), size=n_instances, replace=False)
    query_idx = indices
    query_sample = X[indices]
    return query_idx, query_sample


def make_data_class(X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
    """
    Make classification data for animation

    Args:
    ----------
    X :
        Optional, input feature data X. Must be 2-D

    y :
        Optional, input label data y

    Returns:
    ----------
    x_min :
        min value of x-axis of plot

    x_max :
        max value for x-axis of plot

    y_min :
        min value for y-axis of plot

    y_max :
        max value for y-axis of plot

    xx :
        meshgrid x coords

    yy :
        meshgrid y coords

    X_train :
        Train data features

    X_pool :
        Pool data features

    y_train :
        Train data labels

    y_pool :
        Pool data labels

    """

    # If X,y are not provided create a random classification dataset
    if isinstance(X, type(None)) and isinstance(y, type(None)):
        X, y = make_classification(
            n_samples=200,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            n_classes=3,
            class_sep=1.2,
            random_state=0,
            flip_y=0.00,
        )
    elif not (isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
        print(f"Check if the input arguments X:{X} or y:{y} are not of type np.ndarray")
        return

    # Split data
    X_train, X_pool, y_train, y_pool = train_test_split(X, y, test_size=0.80, random_state=42, stratify=y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, train_size=0.18, random_state=42, stratify=y_train
    )

    # step size in the mesh
    mesh_step = 0.02

    # create a mesh to plot in
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step), np.arange(y_min, y_max, mesh_step))

    return (
        x_min,
        x_max,
        y_min,
        y_max,
        xx,
        yy,
        X_train,
        X_pool,
        y_train,
        y_pool,
        X_test,
        y_test,
    )


def make_data_reg(initial_train_size: int = 3) -> Tuple[np.ndarray, ...]:
    """
    Make data for regression plots

    Args:
    ----------
    initial_train_size :
        The initial train size for your model

    Returns:
    ----------
    X_train:
        Features train data

    y_train:
        Labels train data

    X_pool:
        Features pool data

    y_pool:
        Labels pool data

    X_test:
        Features test data

    y_test:
        Labels test data

    """

    # Sort the data to make plotting easier later
    np.random.seed(37)
    X = np.sort(-10 * np.random.rand(500) + 10)

    # Gaussian Noise
    noise = np.random.normal(0, 1, 500) * 0.4

    # Increase dims for train test split result
    X = np.expand_dims(X, axis=1)

    # Add noise and other functions
    y = np.sin(X).squeeze() + np.cos(X).squeeze() + np.sqrt(X).squeeze() + noise

    # Split for Pool and test
    X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.10, random_state=42, shuffle="False")

    # Split again for train and Pool
    X_train, X_pool, y_train, y_pool = train_test_split(
        X_pool, y_pool, train_size=initial_train_size, random_state=42, shuffle="False"
    )

    return X_test, X_train, X_pool, y_test, y_train, y_pool


def process_uncertainty_result(
    model: BaseEstimator,
    sampling: Callable,
    problem_type: str,
    strat: str,
    n_queries: int,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """
    A function to directly returns the relevant data from the uncertanity results,
    so that it can be used further in other functions

    Args:
    ----------
    model :
        The ML model to be used

    sampling :
        The uncertanity sampling stratergy

    problem_type :
        Classification or Regression

    strat :
        The name of the problem

    n_queries :
        The number of queries/iterations to perform

    X_pool :
        Pool data features

    y_pool :
        Pool data labels

    X_train :
        Train data features

    y_train :
        Train data labels

    Returns:
    ----------
    queries_np_X :
        The queried points

    queries_np_y :
        Corresponding labels for the queried points

    score :
        Accuracy score or RMSE for each queried point

    X_pool :
        Pool data features

    y_pool :
        Pool data labels

    st_dev_pts:
        Standard deviation of each point(Only for regression)

    """
    # Run uncertanity sampling
    uncertanity_data_dict = uncertainty_sampling(
        sampling,
        clone(model),
        strat,
        n_queries=n_queries,
        problem_type=problem_type,
        X_pool=X_pool,
        y_pool=y_pool,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    # Get the quried points and their corresponding labels
    queries_np_X = np.array(uncertanity_data_dict["query_sample"]["X"]).reshape(n_queries, X_train.shape[1])
    queries_np_y = np.array(uncertanity_data_dict["query_sample"]["y"]).reshape(
        n_queries,
    )

    # A list of Pool data and corresponding labels after each query has completed
    X_pool = uncertanity_data_dict["pool"]["X"]
    y_pool = uncertanity_data_dict["pool"]["y"]

    if problem_type == "Classification":
        # Get the accuracy score or RMSE calculated after each query point
        score = uncertanity_data_dict["score"]
        return queries_np_X, queries_np_y, score, X_pool, y_pool
    else:
        st_dev_pts = uncertanity_data_dict["std_dev_pts"]
        score = uncertanity_data_dict["score"]
        return queries_np_X, queries_np_y, score, X_pool, y_pool, st_dev_pts
