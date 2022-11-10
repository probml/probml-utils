# to show output from the 'tests', run with 
# pytest logreg_flax_test.py  -rP

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
#import jax.debug
import itertools
from itertools import repeat
from time import time
import chex
import typing

import jax
from typing import Any, Callable, Sequence
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn
import flax

import jaxopt
import optax

import sklearn.datasets
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression

from logreg_flax import *
#jax.config.update("jax_enable_x64", True) # jaxopt.lbfgs uses float32

def print_probs(probs):
    str = ['{:0.3f}'.format(p) for p in probs]
    print(str)

def make_iris_data():
    iris = sklearn.datasets.load_iris()
    X = iris["data"]
    #y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0'
    y = iris["target"]
    nclasses = len(np.unique(y)) # 3
    ndata, ndim = X.shape  # 150, 4
    key = jr.PRNGKey(0)
    noise = jr.normal(key, (ndata, ndim)) * 2.0
    X = X + noise # add noise to make the classes less separable
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X, y

def make_data(seed, n_samples, class_sep, n_features):
    X, y = sklearn.datasets.make_classification(n_samples=n_samples, n_features=n_features,  n_informative=5,
    n_redundant=5, n_repeated=0, n_classes=10, n_clusters_per_class=1, weights=None, flip_y=0.01,
    class_sep=class_sep, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=seed)
    return X, y

def compute_mle(X, y):
    # We set C to a large number to turn off regularization.
    # We don't fit the bias term to simplify the comparison below.
    log_reg = LogisticRegression(solver="lbfgs", C=1e5, fit_intercept=True)
    log_reg.fit(X, y)
    W_mle = log_reg.coef_ # (nclasses, ndim)
    b_mle = log_reg.intercept_ # (nclasses,)
    true_probs = log_reg.predict_proba(X)
    return true_probs, W_mle, b_mle

############

def test_inference():
    # test inference at the MLE params
    X, y = make_iris_data()
    true_probs, W_mle, b_mle = compute_mle(X, y)
    nclasses, ndim = W_mle.shape
    key = jr.PRNGKey(0)
    model = LogReg(key,  nclasses, W_init=W_mle.T, b_init=b_mle)
    probs = np.array(model.predict(X))
    assert np.allclose(probs, true_probs, atol=1e-2)

def compare_training(X, y, name, tol=0.02):
    true_probs, W_mle, b_mle = compute_mle(X, y)
    nclasses, ndim = W_mle.shape
    key = jr.PRNGKey(0)
    model = LogReg(key, nclasses, max_iter=500, l2reg=1e-5) 
    model.fit(X, y)
    probs = np.array(model.predict(X))
    delta = np.max(true_probs - probs)
    print('dataset ', name)
    print('max difference in predicted probabilities', delta)
    print('truth'); print_probs(true_probs[0])
    print('pred'); print_probs(probs[0])
    assert (delta < tol)

def test_training_iris():
    X, y = make_iris_data()
    compare_training(X, y, 'iris', 0.02)

def test_training_blobs():
    X, y = make_data(0, n_samples=1000, class_sep=1, n_features=10) 
    compare_training(X, y, 'blobs', 0.02)



def skip_test_objectives():
    # compare logprior to the l2 regularizer
    X, y = make_iris_data()
    ndata, ndim = X.shape
    nclasses = 3
    key = jr.PRNGKey(0)
    model = LogReg(key, nclasses, max_iter=2, l2reg=1e-5) 
    model.fit(X, y) # we need to fit the model to populate the params field
    params = model.params
    leaves, _ = jax.tree_util.tree_flatten(params)
    flat_params = jnp.concatenate([jnp.ravel(a) for a in leaves])
    nparams = len(flat_params)
    logits = model.network.apply(params, X)
    l2reg = 0.1
    sigma = np.sqrt(1/l2reg)
    l1 = loss_from_logits(params, l2reg, logits, y)
    l2 = loglikelihood_fn(params, model.network, X, y)
    l3 = logprior_fn(params, sigma)
    Z = sigma*np.sqrt(2*np.pi)
    # log p(w) = sum_i log N(wi | 0, sigma)  = sum_i [-log Z_i - 0.5*l2reg*w_i^2]
    assert np.allclose(-l1 -nparams*np.log(Z), l2+l3)

##########

def fit_pipeline_sklearn(key, X, Y):
    classifier = Pipeline([
            ('standardscaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2)), 
            ('logreg', LogisticRegression(random_state=0, max_iter=500, C=1e5))])
    classifier.fit(np.array(X), np.array(Y))
    return classifier

def fit_pipeline_logreg(key, X, Y):
    nclasses  = len(np.unique(Y))
    classifier = Pipeline([
            ('standardscaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2)), 
            ('logreg', LogReg(key, nclasses, max_iter=500, l2reg=1e-5))])
    classifier.fit(np.array(X), np.array(Y))
    return classifier
    

def compare_pipeline(X, y, name, tol=0.02):
    key = jr.PRNGKey(0)
    clf = fit_pipeline_sklearn(key, X, y)
    true_probs = clf.predict_proba(X)
    model = fit_pipeline_logreg(key, X, y)
    probs = np.array(model.predict(X))
    delta = np.max(true_probs - probs)
    print('data ', name)
    print('max difference in predicted probs {:.3f}'.format(delta))
    print('truth: ', true_probs[0])
    print('pred: ', probs[0])
    assert delta < tol

def test_pipeline_iris():
    X, y = make_iris_data()
    compare_pipeline(X, y, 'iris', 0.02)
 
def test_pipeline_blobs():
    X, y = make_data(0, n_samples=1000, class_sep=1, n_features=10) 
    compare_pipeline(X, y, 'blobs', 0.1) # much less accurate!



#########

def compare_optimizer(optimizer, name=None, batch_size=None, max_iter=5000, tol=0.02):
    X, y = make_iris_data()
    true_probs, W_mle, b_mle = compute_mle(X, y)
    nclasses, ndim = W_mle.shape
    key = jr.PRNGKey(0)
    l2reg = 1e-5
    model = LogReg(key, nclasses, max_iter=max_iter, l2reg=l2reg, optimizer=optimizer, batch_size=batch_size)  
    model.fit(X, y)
    probs = np.array(model.predict(X))
    error = np.max(true_probs - probs)
    print('method {:s}, max deviation from true probs {:.3f}'.format(name, error))
    print('truth: ', true_probs[0])
    print('pred: ', probs[0])
    assert (error < tol)


def test_bfgs():
    compare_optimizer("lbfgs", name= "lbfgs, bs=N", batch_size=0)

def test_armijo_full_batch():
    compare_optimizer("armijo", name="armijo, bs=N", batch_size=0)


def test_adam_full_batch_lr2():
    compare_optimizer(optax.adam(1e-2), name="adam 1e-2, bs=N", batch_size=0)

# These tests fail at reasonable tolerance

def test_armijo_minibatch():
    compare_optimizer("armijo", name="armijo, bs=32", batch_size=32, tol=0.25)

def test_adam_mini_batch_lr2():
    compare_optimizer(optax.adam(1e-2), name="adam 1e-2, bs=32", batch_size=32, tol=0.1)
