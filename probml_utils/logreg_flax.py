# Logistic regression using flax, optax and jaxopt
# Since the objective is convex, we should be able to get good results using
# BFGS or first order methods with automatic (Armijo) step size tuning.

from functools import partial
import matplotlib.pyplot as plt
import numpy as np
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
import distrax
from jaxopt import OptaxSolver
import tensorflow as tf

from sklearn.base import ClassifierMixin

logistic_loss = jax.vmap(jaxopt.loss.multiclass_logistic_loss)

def regularizer(params, l2reg):
    sqnorm = jaxopt.tree_util.tree_l2_norm(params, squared=True)
    return 0.5 * l2reg * sqnorm

def loss_from_logits(params, l2reg, logits, labels):
    mean_loss = jnp.mean(logistic_loss(labels, logits))
    return mean_loss + regularizer(params, l2reg)


def loglikelihood_fn(params, model, X, y):
    # 1/N sum_n log p(yn | xn, params)
    logits = model.apply(params, X)
    return jnp.mean(distrax.Categorical(logits).log_prob(y))

def logprior_fn(params, sigma):
    # log p(params)
    leaves, _ = jax.tree_util.tree_flatten(params)
    flat_params = jnp.concatenate([jnp.ravel(a) for a in leaves])
    return jnp.sum(distrax.Normal(0, sigma).log_prob(flat_params))

@partial(jax.jit, static_argnames=["network"])
def objective(params, data, network, prior_sigma, ntrain): 
    # objective = -1/N [ (sum_n log p(yn|xn, theta)) + log p(theta) ]
    X, y = data["X"], data["y"]
    logjoint = loglikelihood_fn(params, network, X, y) + (1/ntrain)*logprior_fn(params, prior_sigma)
    return -logjoint


class LogRegNetwork(nn.Module):
    nclasses: int
    W_init_fn: Any
    b_init_fn: Any

    @nn.compact
    def __call__(self, x):
        if self.W_init_fn is not None:
            logits = nn.Dense(self.nclasses, kernel_init=self.W_init_fn, bias_init=self.b_init_fn)(x)
        else:
            logits = nn.Dense(self.nclasses)(x)
        return logits

class LogReg(ClassifierMixin):
    def __init__(self, key, nclasses, *,  l2reg=1e-5,
                optimizer = 'lbfgs', batch_size=0, max_iter=500, 
                 W_init=None, b_init=None):
        # optimizer is {'lbfgs', 'polyak', 'armijo'} or an optax object
        self.nclasses = nclasses
        if W_init is not None: # specify initial parameters by hand
            W_init_fn = lambda key, shape, dtype: W_init # (D,C)
            b_init_fn = lambda key, shape, dtype: b_init # (C)
            ninputs = W_init.shape[0]
            self.network = LogRegNetwork(nclasses, W_init_fn, b_init_fn)
            x = jr.normal(key, (ninputs,)) # single random input 
            self.params = self.network.init(key, x)
        else:
            self.network = LogRegNetwork(nclasses, None, None)
            self.params = None # must call fit to infer size of input
        self.optimization_results = None
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.l2reg = l2reg
        self.key = key
  
    
    def predict(self, inputs):
        return jax.nn.softmax(self.network.apply(self.params, inputs))

    def fit(self, X, y):
        self.params = self.network.init(self.key, X[0])
        N = X.shape[0]
        if (self.batch_size == 0) or (self.batch_size == N):
            return self.fit_batch(self.key, X, y)
        else:
            return self.fit_minibatch(self.key, X, y)

    def fit_batch(self, key, X, y):
        del key
        # This version is fully deterministic
        sigma = np.sqrt(1/self.l2reg)
        N = X.shape[0]
        data = {"X": X, "y": y}
        def loss_fn(params):
            return objective(params=params, data=data,  network=self.network,  prior_sigma=sigma, ntrain=N)

        if isinstance(self.optimizer, str) and (self.optimizer.lower() == "lbfgs"):
            solver = jaxopt.LBFGS(fun=loss_fn, maxiter=self.max_iter)
        elif isinstance(self.optimizer, str) and (self.optimizer.lower() == "polyak"):
            solver = jaxopt.PolyakSGD(fun=loss_fn, maxiter=self.max_iter)
        elif isinstance(self.optimizer, str) and (self.optimizer.lower() == "armijo"):
            solver = jaxopt.ArmijoSGD(fun=loss_fn, maxiter=self.max_iter)
        else:
            solver = OptaxSolver(opt=self.optimizer, fun=loss_fn, maxiter=self.max_iter)

        res = solver.run(self.params)
        self.params = res.params

    def fit_minibatch(self, key, X, y):
        del key
        # https://jaxopt.github.io/stable/auto_examples/deep_learning/flax_resnet.html
        # https://github.com/blackjax-devs/blackjax/discussions/360#discussioncomment-3756412
        sigma = np.sqrt(1/self.l2reg)
        N, B = X.shape[0], self.batch_size
        def loss_fn(params, data):
            return objective(params=params, data=data,  network=self.network,  prior_sigma=sigma, ntrain=N)

        # Convert dataset into a stream of minibatches (for stochasitc optimizers)
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset?version=nightly#from_tensor_slices
        ds = tf.data.Dataset.from_tensor_slices({"X": X, "y": y})
        # https://jaxopt.github.io/stable/auto_examples/deep_learning/haiku_image_classif.htm
        ds = ds.cache().repeat()
        ds = ds.shuffle(10 * self.batch_size, seed=0) # how use jax key?
        ds = ds.batch(self.batch_size)
        iterator = ds.as_numpy_iterator()

        if isinstance(self.optimizer, str) and (self.optimizer.lower() == "lbfgs"):
            solver = jaxopt.LBFGS(fun=loss_fn, maxiter=self.max_iter)
        elif isinstance(self.optimizer, str) and (self.optimizer.lower() == "polyak"):
            solver = jaxopt.PolyakSGD(fun=loss_fn, maxiter=self.max_iter)
        elif isinstance(self.optimizer, str) and (self.optimizer.lower() == "armijo"):
            solver = jaxopt.ArmijoSGD(fun=loss_fn, maxiter=self.max_iter)
        else:
            solver = OptaxSolver(opt=self.optimizer, fun=loss_fn, maxiter=self.max_iter)
    
        res = solver.run_iterator(self.params, iterator=iterator)
        self.params = res.params


