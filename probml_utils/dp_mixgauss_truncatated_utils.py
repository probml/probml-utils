# Transformed from
# https://github.com/ericmjl/dl-workshop/blob/master/src/dl_workshop/gaussian_mixture.py
# change the code to work with the current jax

import jax.numpy as jnp
from jax.scipy import stats


def loglike_one_component(component_weight, component_mu, log_component_scale, datum):
    """Log likelihood of datum under one component of the mixture.
    Defined as the log likelihood of observing that datum from the component
    (i.e. log of component probability)
    added to the log likelihood of observing that datum
    under the Gaussian that belongs to that component.
    :param component_weight: Component weight, a scalar value between 0 and 1.
    :param component_mu: A scalar value.
    :param log_component_scale: A scalar value.
        Gets exponentiated before being passed into norm.logpdf.
    :returns: A scalar.
    """
    component_scale = jnp.exp(log_component_scale)
    return jnp.log(component_weight) + stats.norm.logpdf(datum, loc=component_mu, scale=component_scale)


def normalize_weights(weights):
    """Normalize a weights vector to sum to 1."""
    return weights / jnp.sum(weights)


from functools import partial
from jax.scipy.special import logsumexp
from jax import vmap


def loglike_across_components(log_component_weights, component_mus, 
                              log_component_scales, datum):
    """Log likelihood of datum under all components of the mixture."""
    component_weights = normalize_weights(jnp.exp(log_component_weights))
    loglike_components = vmap(partial(loglike_one_component, datum=datum))(
        component_weights, component_mus, log_component_scales)
    return logsumexp(loglike_components)


def mixture_loglike(log_component_weights, component_mus, 
                    log_component_scales, data):
    """Log likelihood of data (not datum!) under all components of the mixture."""
    ll_per_data = vmap(partial(loglike_across_components, log_component_weights, 
                               component_mus, log_component_scales,))(data)
    return jnp.sum(ll_per_data)


from jax.scipy.stats import norm


def plot_component_norm_pdfs(log_component_weights, component_mus, 
                             log_component_scales, xmin, xmax, ax, title):
    component_weights = normalize_weights(jnp.exp(log_component_weights))
    component_scales = jnp.exp(log_component_scales)
    x = jnp.linspace(xmin, xmax, 1000).reshape(-1, 1)
    pdfs = component_weights * norm.pdf(x, loc=component_mus, scale=component_scales)
    for component in range(pdfs.shape[1]):
        ax.plot(x, pdfs[:, component])
    ax.set_title(title)


def get_loss(state, get_params_func, loss_func, data):
    params = get_params_func(state)
    loss_score = loss_func(params, data)
    return loss_score


import matplotlib.pyplot as plt
from celluloid import Camera


def animate_training(params_for_plotting, interval, data_mixture):
    """Animation function for mixture likelihood."""
    log_component_weights_history = params_for_plotting['log_component_weight']
    component_mus_history = params_for_plotting['component_mus']
    log_component_scales_history = params_for_plotting['log_component_scale']
    fig, ax = plt.subplots()
    cam = Camera(fig)
    for w, m, s in zip(log_component_weights_history[::interval], 
                       component_mus_history[::interval], 
                       log_component_scales_history[::interval]):
        ax.hist(data_mixture, bins=40, density=True, color="blue")
        plot_component_norm_pdfs(w, m, s, xmin=-20, xmax=20, ax=ax, title=None)
        cam.snap()
    animation = cam.animate()
    return animation


from jax import lax


def stick_breaking_weights(beta_draws):
    """Return weights from a stick breaking process.
    :param beta_draws: i.i.d draws from a Beta distribution.
        This should be a row vector.
    """
    def weighting(occupied_probability, beta_i):
        """
        :param occupied_probability: The cumulative occupied probability taken up.
        :param beta_i: Current value of beta to consider.
        """
        weight = (1 - occupied_probability) * beta_i
        return occupied_probability + weight, weight
    occupied_probability, weights = lax.scan(weighting, jnp.array(0.0), beta_draws)
    weights = weights / jnp.sum(weights)
    return occupied_probability, weights


from jax import random


def beta_draw_from_weights(weights):
    def beta_from_w(accounted_probability, weights_i):
        """
        :param accounted_probability: The cumulative probability acounted for.
        :param weights_i: Current value of weights to consider.
        """
        denominator = 1 - accounted_probability
        log_denominator = jnp.log(denominator)
        log_beta_i = jnp.log(weights_i) - log_denominator
        newly_accounted_probability = accounted_probability + weights_i
        return newly_accounted_probability, jnp.exp(log_beta_i)
    final, betas = lax.scan(beta_from_w, jnp.array(0.0), weights)
    return final, betas


def component_probs_loglike(log_component_probs, log_concentration, num_components):
    """Evaluate log likelihood of probability vector under Dirichlet process.
    :param log_component_probs: A vector.
    :param log_concentration: Real-valued scalar.
    :param num_compnents: Scalar integer.
    """
    concentration = jnp.exp(log_concentration)
    component_probs = normalize_weights(jnp.exp(log_component_probs))
    _, beta_draws = beta_draw_from_weights(component_probs)
    eval_draws = beta_draws[:num_components]
    return jnp.sum(stats.beta.logpdf(x=eval_draws, a=1, b=concentration))
