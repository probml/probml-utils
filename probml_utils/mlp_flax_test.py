# to show output from the 'tests', run with 
# pytest skax_test_mlp.py  -rP

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

from probml_utils.mlp_flax import MLPNetwork, NeuralNetClassifier

def print_vec(probs):
    str = ['{:0.3f}'.format(p) for p in probs]
    print(str)

### Create synthetic dataset from GMM, compare MLP predictions (with different optimizers) to Bayes optimal

@chex.dataclass
class GenParams:
    nclasses: int
    nfeatures: int
    prior: chex.Array
    mus: chex.Array # (C,D)
    Sigmas: chex.Array #(C,D,D)

def make_params(key, nclasses, nfeatures, scale_factor=1):
    mus = jr.normal(key, (nclasses, nfeatures)) # (C,D)
    # shared covariance -> linearly separable
    #Sigma = scale_factor * jnp.eye(nfeatures)
    #Sigmas = jnp.array([Sigma for _ in range(nclasses)]) # (C,D,D)
    # diagonal covariance -> nonlinear decision boundaries
    sigmas = jr.uniform(key, shape=(nclasses, nfeatures), minval=0.5, maxval=5)
    Sigmas = jnp.array([scale_factor*jnp.diag(sigmas[y]) for y in range(nclasses)])
    prior = jnp.ones(nclasses)/nclasses
    return GenParams(nclasses=nclasses, nfeatures=nfeatures, prior=prior, mus=mus, Sigmas=Sigmas)

def sample_data(key, params, nsamples):
    y = jr.categorical(key, logits=jnp.log(params.prior), shape=(nsamples,))
    X = jr.multivariate_normal(key, params.mus[y], params.Sigmas[y])
    return X, y

def predict_bayes(X, params):
    def lik_fn(y):
        return jsp.stats.multivariate_normal.pdf(X, params.mus[y], params.Sigmas[y])
    liks = vmap(lik_fn)(jnp.arange(params.nclasses)) # liks(k,n)=p(X(n,:) | y=k)
    joint = jnp.einsum('kn,k -> nk', liks, params.prior) # joint(n,k) = liks(k,n) * prior(k)
    norm = joint.sum(axis=1) # norm(n)  = sum_k joint(n,k) = p(X(n,:)
    post = joint / jnp.expand_dims(norm, axis=1) # post(n,k) = p(y = k | xn)
    return post

def compare_bayes(optimizer, name, nhidden, scale_factor):
    nclasses = 4
    key = jr.PRNGKey(0)
    key, subkey = jr.split(key)
    params = make_params(subkey, nclasses=nclasses, nfeatures=10, scale_factor=scale_factor)
    key, subkey = jr.split(key)
    Xtrain, ytrain = sample_data(subkey, params, nsamples=1000)
    key, subkey = jr.split(key)
    Xtest, ytest = sample_data(subkey, params, nsamples=1000)

    yprobs_train_bayes = predict_bayes(Xtrain, params)
    yprobs_test_bayes = predict_bayes(Xtest, params)

    ypred_train_bayes = jnp.argmax(yprobs_train_bayes, axis=1)
    error_rate_train_bayes = jnp.sum(ypred_train_bayes != ytrain) / len(ytrain)

    ypred_test_bayes = jnp.argmax(yprobs_test_bayes, axis=1)
    error_rate_test_bayes = jnp.sum(ypred_test_bayes != ytest) / len(ytest)

    nhidden = nhidden + (nclasses,) # set nhidden() to get logistic regression
    network = MLPNetwork(nhidden)
    mlp = NeuralNetClassifier(network, key, nclasses, l2reg=1e-5, optimizer = optimizer, 
            batch_size=32, num_epochs=30, print_every=0)  
    mlp.fit(Xtrain, ytrain)

    yprobs_train_mlp = np.array(mlp.predict(Xtrain))
    yprobs_test_mlp = np.array(mlp.predict(Xtest))

    ypred_train_mlp = jnp.argmax(yprobs_train_mlp, axis=1)
    error_rate_train_mlp = jnp.sum(ypred_train_mlp != ytrain) / len(ytrain)

    ypred_test_mlp = jnp.argmax(yprobs_test_mlp, axis=1)
    error_rate_test_mlp = jnp.sum(ypred_test_mlp != ytest) / len(ytest)
    
    delta_train = jnp.max(yprobs_train_bayes - yprobs_train_mlp)
    delta_test = jnp.max(yprobs_test_bayes - yprobs_test_mlp)

    print('Evaluating training method {:s} on model with {} hidden layers'.format(name, nhidden))
    print('Train error rate {:.3f} (Bayes {:.3f}), Test error rate {:.3f} (Bayes {:.3f})'.format(
        error_rate_train_mlp, error_rate_train_bayes, error_rate_test_mlp, error_rate_test_bayes))
    #print('Max diff in probs from Bayes: train {:.3f}, test {:.3f}'.format(
    #    delta_train, delta_test))
    print('\n')

def test_mlp_vs_bayes_sf1():
    compare_bayes(optax.adam(1e-3), "adam(1e-3)", nhidden=(), scale_factor=1)
    compare_bayes("adam+warmup", "adam+warmup", nhidden=(), scale_factor=1)

    compare_bayes(optax.adam(1e-3), "adam(1e-3)", nhidden=(10,), scale_factor=1)
    compare_bayes("adam+warmup", "adam+warmup", nhidden=(10,), scale_factor=1)

    compare_bayes(optax.adam(1e-3), "adam(1e-3)", nhidden=(10,10), scale_factor=1)
    compare_bayes("adam+warmup", "adam+warmup", nhidden=(10,10), scale_factor=1)


def test_mlp_vs_bayes_sf5():
   # scale_factor = 5 means the class conditional densities have higher variance (more overlap)

    compare_bayes(optax.adam(1e-3), "adam(1e-3)", nhidden=(), scale_factor=5)
    compare_bayes("adam+warmup", "adam+warmup", nhidden=(), scale_factor=5)

    compare_bayes(optax.adam(1e-3), "adam(1e-3)", nhidden=(10,), scale_factor=5)
    compare_bayes("adam+warmup", "adam+warmup", nhidden=(10,), scale_factor=5)

    compare_bayes(optax.adam(1e-3), "adam(1e-3)", nhidden=(10,10), scale_factor=5)
    compare_bayes("adam+warmup", "adam+warmup", nhidden=(10,10), scale_factor=5)
