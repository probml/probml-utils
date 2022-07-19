from typing import Tuple, Sequence
from functools import partial
import os

import jax
import jax.numpy as jnp
import numpy as np

try:
    import flax
except ModuleNotFoundError:
    os.system("pip install -qq flax")
    import flax
import flax.linen as nn
from flax.training import train_state

try:
    import optax
except ModuleNotFoundError:
    os.system("pip install -qq optax")
    import optax


class Encoder(nn.Module):
    latent_dim: int
    hidden_dims: Sequence[int] = (32, 64, 128, 256, 512)

    @nn.compact
    def __call__(self, X, training):
        for hidden_dim in self.hidden_dims:
            X = nn.Conv(hidden_dim, (3, 3), strides=2, padding=1)(X)
            X = nn.BatchNorm(use_running_average=not training)(X)
            X = jax.nn.relu(X)

        X = X.reshape((-1, np.prod(X.shape[-3:])))
        mu = nn.Dense(self.latent_dim)(X)
        logvar = nn.Dense(self.latent_dim)(X)

        return mu, logvar


class Decoder(nn.Module):
    output_dim: Tuple[int, int, int]
    hidden_dims: Sequence[int] = (32, 64, 128, 256, 512)

    @nn.compact
    def __call__(self, X, training):
        H, W, C = self.output_dim

        # TODO: relax this restriction
        factor = 2**len(self.hidden_dims)
        assert H % factor == W % factor == 0, f"output_dim must be a multiple of {factor}"
        H, W = H // factor, W // factor

        X = nn.Dense(H * W * self.hidden_dims[-1])(X)
        X = jax.nn.relu(X)
        X = X.reshape((-1, H, W, self.hidden_dims[-1]))

        for hidden_dim in reversed(self.hidden_dims[:-1]):
            X = nn.ConvTranspose(hidden_dim, (3, 3), strides=(2, 2), padding=((1, 2), (1, 2)))(X)
            X = nn.BatchNorm(use_running_average=not training)(X)
            X = jax.nn.relu(X)

        X = nn.ConvTranspose(C, (3, 3), strides=(2, 2), padding=((1, 2), (1, 2)))(X)
        X = jax.nn.sigmoid(X)

        return X


def reparameterize(key, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(key, logvar.shape)
    return mean + eps * std


class VAE(nn.Module):
    variational: bool
    latent_dim: int
    output_dim: Tuple[int, int, int]

    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.output_dim)

    def __call__(self, key, X, training):
        mean, logvar = self.encoder(X, training)
        if self.variational:
            Z = reparameterize(key, mean, logvar)
        else:
            Z = mean

        recon = self.decoder(Z, training)
        return recon, mean, logvar

    def decode(self, Z, training):
        return self.decoder(Z, training)


class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict[str, jnp.ndarray]
    beta: float


def create_train_state(key, variational, beta, latent_dim, learning_rate, specimen):
    vae = VAE(variational, latent_dim, specimen.shape)
    key_dummy = jax.random.PRNGKey(42)
    (recon, _, _), variables = vae.init_with_output(key, key_dummy, specimen, True)
    assert recon.shape[-3:] == specimen.shape, f"{recon.shape} = recon.shape != specimen.shape = {specimen.shape}"
    tx = optax.adam(learning_rate)
    state = TrainState.create(
        apply_fn=vae.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
        beta=beta,
    )

    return state


@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.jit
def train_step(state, key, image):
    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        (recon, mean, logvar), new_model_state = state.apply_fn(variables, key, image, True, mutable=["batch_stats"])
        loss = jnp.sum((recon - image) ** 2) + state.beta * jnp.sum(kl_divergence(mean, logvar))
        return loss.sum(), new_model_state

    (loss, new_model_state), grads = loss_fn(state.params)

    state = state.apply_gradients(grads=grads, batch_stats=new_model_state["batch_stats"])

    return state, loss


@jax.jit
def test_step(state, key, image):
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    recon, mean, logvar = state.apply_fn(variables, key, image, False)

    return recon, mean, logvar


@jax.jit
def decode(state, Z):
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    decoded = state.apply_fn(variables, Z, False, method=VAE.decode)

    return decoded
