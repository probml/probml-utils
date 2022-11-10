

# Compare logistic regression in sklearn to our MLP model with no hidden layers on some small synthetic data
# We find comparable results in predictive probabilites provided we use learning rate warmup.

from functools import partial
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3)
import scipy.stats
import einops
import matplotlib
from functools import partial
from collections import namedtuple
import jax
import jax.random as jr
import jax.numpy as jnp
from jax import vmap, grad, jit
import jax.scipy as jsp
import itertools
from itertools import repeat
from time import time
import chex
import typing

import jax
from typing import Any, Callable, Sequence
from jax import lax, random, numpy as jnp
from flax import linen as nn
import flax

import jaxopt
import optax

import sklearn.datasets
import sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


from probml_utils.mlp_flax import MLPNetwork, NeuralNetClassifier

def make_data(seed, n_samples, class_sep, n_features):
    X, y = sklearn.datasets.make_classification(n_samples=n_samples, n_features=n_features,  n_informative=5,
    n_redundant=5, n_repeated=0, n_classes=10, n_clusters_per_class=1, weights=None, flip_y=0.01,
    class_sep=class_sep, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=seed)
    return X, y

def fit_predict_logreg(Xtrain, ytrain, Xtest, ytest, l2reg):
    # Use sklearn to fit logistic regression model
    classifier = Pipeline([
            ('standardscaler', StandardScaler()), 
            ('logreg', LogisticRegression(random_state=0, max_iter=100, C=1/l2reg))])
    classifier.fit(Xtrain, ytrain)
    train_probs = classifier.predict_proba(Xtrain)
    test_probs = classifier.predict_proba(Xtest)
    return train_probs, test_probs

def compare_probs(logreg_probs, probs, labels):
    delta = np.max(logreg_probs - probs)
    logreg_pred = np.argmax(logreg_probs, axis=1)
    pred = np.argmax(probs, axis=1)
    logreg_error_rate = np.mean(logreg_pred != labels)
    error_rate = np.mean(pred != labels)
    return delta, logreg_error_rate, error_rate

def compare_logreg(optimizer, name=None, batch_size=None, num_epochs=30,
                   n_samples=1000, class_sep=1, n_features=10):
    key = jr.PRNGKey(0)
    l2reg = 1e-5
    X_train, y_train = make_data(0, n_samples, class_sep, n_features)
    X_test, y_test = make_data(1, 1000, class_sep, n_features)
    nclasses = len(np.unique(y_train))
    train_probs_logreg, test_probs_logreg = fit_predict_logreg(X_train, y_train, X_test, y_test, l2reg)
    
    #network = MLPNetwork((5, nclasses,))
    network = MLPNetwork((nclasses,)) # no hidden layers == logistic regression
    model = NeuralNetClassifier(network, key, nclasses, l2reg=l2reg, optimizer = optimizer, 
            batch_size=batch_size, num_epochs=num_epochs, print_every=1)  
    model.fit(X_train, y_train)
    train_probs = np.array(model.predict(X_train))
    test_probs = np.array(model.predict(X_test))

    train_delta, train_logreg_error_rate, train_error_rate = compare_probs(train_probs_logreg, train_probs, y_train)
    test_delta, test_logreg_error_rate, test_error_rate = compare_probs(test_probs_logreg, test_probs, y_test)
    print('max difference in train probabilities from logreg to {:s} is {:.3f}'.format(name, train_delta))
    print('misclassification rates: logreg train = {:.3f}, model train = {:.3f}'.format(
            train_logreg_error_rate, train_error_rate))
    print('misclassification rates: logreg test = {:.3f}, model test = {:.3f}'.format(
        test_logreg_error_rate, test_error_rate))


compare_logreg(optax.adam(1e-3), name="adam 1e-3, bs=32", batch_size=32)

compare_logreg("adam+warmup", name="adam+warmup, bs=32", batch_size=32)