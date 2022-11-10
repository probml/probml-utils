


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import einops

from functools import partial
import jax
import jax.random as jr
import jax.numpy as jnp
from itertools import repeat

import jax
from typing import Any, Callable, Sequence
from jax import lax, random, numpy as jnp
from flax import linen as nn
import flax
from flax.training import train_state

import optax
import distrax




class LogRegNetwork(nn.Module):
    nclasses: int

    @nn.compact
    def __call__(self, x):
        logits = nn.Dense(self.nclasses)(x)
        return logits

class MLPNetwork(nn.Module):
  nfeatures_per_layer: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    nlayers = len(self.nfeatures_per_layer)
    for i, feat in enumerate(self.nfeatures_per_layer):
      x = nn.Dense(feat, name=f'layers_{i}')(x)
      if i != (nlayers - 1):
        #x = nn.relu(x)
        x = nn.gelu(x)
    return x


def logprior_fn(params, sigma):
    # log p(params)
    leaves, _ = jax.tree_util.tree_flatten(params)
    flat_params = jnp.concatenate([jnp.ravel(a) for a in leaves])
    return jnp.sum(distrax.Normal(0, sigma).log_prob(flat_params))


@partial(jax.jit, static_argnums=(1,2))
def get_batch_train_ixs(key, num_train, batch_size):
    # return indices of training set in a random order
    # Based on https://github.com/google/flax/blob/main/examples/mnist/train.py#L74
    steps_per_epoch = num_train // batch_size
    batch_ixs = jax.random.permutation(key, num_train)
    batch_ixs = batch_ixs[:steps_per_epoch * batch_size]
    batch_ixs = batch_ixs.reshape(steps_per_epoch, batch_size)
    return batch_ixs


class NeuralNetClassifier:
    def __init__(self, network, key, nclasses, *,  l2reg=1e-5, standardize = True,
                optimizer = 'lbfgs', batch_size=128, max_iter=100, num_epochs=10, print_every=0):
        # optimizer is one of {'adam+warmup'} or an optax object
        self.nclasses = nclasses
        self.network = network
        self.standardize = standardize
        self.max_iter = max_iter
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.l2reg = l2reg
        self.print_every = print_every
        self.params = None # must first call fit
        self.key = key

    def predict(self, X):
        if self.params is None:
            raise ValueError('need to call fit before predict')
        if self.standardize:
            X = X - self.mean
            X = X / self.std
        return jax.nn.softmax(self.network.apply(self.params, X))

    def fit(self, X, y):
        """Fit model. We assume y is (N) integer labels, not one-hot."""
        if self.standardize:
            self.mean = jnp.mean(X, axis=0)
            self.std = jnp.std(X, axis=0) + 1e-5 
            X = X - self.mean
            X = X / self.std
        if self.params is None: # initialize model parameters
            nfeatures = X.shape[1]
            x = jr.normal(self.key, (nfeatures,)) # single random input 
            self.params = self.network.init(self.key, x) 
        ntrain = X.shape[0]
        if isinstance(self.optimizer, str) and (self.optimizer.lower() == "adam+warmup"):
            total_steps = self.num_epochs*(ntrain//self.batch_size)  
            warmup_cosine_decay_scheduler = optax.warmup_cosine_decay_schedule(
                init_value=1e-3, peak_value=1e-1, warmup_steps=int(total_steps*0.1),
                decay_steps=total_steps, end_value=1e-3)
            self.optimizer = optax.adam(learning_rate=warmup_cosine_decay_scheduler)
            return self.fit_optax(self.key, X, y)
        else:
            return self.fit_optax(self.key, X, y)


    def fit_optax(self, key, X, y): 
        # based on https://github.com/google/flax/blob/main/examples/mnist/train.py
        ntrain = X.shape[0] # full dataset

        @jax.jit
        def train_step(state, Xb, yb):
            # loss = -1/N [ (sum_n log p(yn|xn, theta)) + log p(theta) ]
            # We estimate this from a minibatch.
            # We assume yb is integer, not one-hot.
            def loss_fn(params):
                logits = state.apply_fn({'params': params}, Xb)
                loglik = jnp.mean(distrax.Categorical(logits).log_prob(yb))
                sigma = np.sqrt(1/self.l2reg)
                logjoint = loglik + (1/ntrain)*logprior_fn(params, sigma)
                loss = -logjoint
                return loss, logits
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, logits), grads = grad_fn(state.params)
            accuracy = jnp.mean(jnp.argmax(logits, -1) == yb)
            return grads, loss, accuracy

        def train_epoch(key, state):
            key, sub_key = jr.split(key)
            batch_ixs = get_batch_train_ixs(sub_key, ntrain, self.batch_size)  # shuffles 
            epoch_loss = []
            epoch_accuracy = []
            for batch_ix in batch_ixs:
                X_batch, y_batch = X[batch_ix], y[batch_ix]
                grads, loss, accuracy = train_step(state, X_batch, y_batch)
                state = state.apply_gradients(grads=grads)
                epoch_loss.append(loss)
                epoch_accuracy.append(accuracy)
            train_loss = np.mean(epoch_loss)
            train_accuracy = np.mean(epoch_accuracy)
            return state, train_loss, train_accuracy

        # main loop
        state = train_state.TrainState.create(
            apply_fn=self.network.apply, params=self.params['params'], tx=self.optimizer)
        for epoch in range(self.num_epochs):
            key, sub_key = jr.split(key)
            state, train_loss, train_accuracy = train_epoch(sub_key, state)
            if (self.print_every > 0) and (epoch % self.print_every == 0):
                print('epoch {:d}, train loss {:0.3f}, train accuracy {:0.3f}'.format(
                    epoch, train_loss, train_accuracy))

        self.params = {'params': state.params}

        
      
        
