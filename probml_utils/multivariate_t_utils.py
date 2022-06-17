import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax.numpy.linalg import slogdet, solve


@jax.jit
def log_p_of_multi_t(x, nu, mu, Sigma):
    """
    Computing the logarithm of probability density of the multivariate T distribution,
    https://en.wikipedia.org/wiki/Multivariate_t-distribution
    ---------------------------------------------------------
    x: array(dim)
        Data point that we want to evaluate log pdf at
    nu: int
        Degree of freedom of the multivariate T distribution
    mu: array(dim)
        Location parameter of the multivariate T distribution
    Sigma: array(dim, dim) 
        Positive-definite real scale matrix of the multivariate T distribution
    --------------------------------------------------------------------------
    * float
        Log probability of the multivariate T distribution at x
    """
    dim = mu.shape[0]
    # Logarithm of the normalizing constant
    l0 = gammaln((nu+dim)/2.0) - (gammaln(nu/2.0) + dim/2.0*(jnp.log(nu)+jnp.log(np.pi)) + slogdet(Sigma)[1])
    # Logarithm of the unnormalized pdf
    l1 = -(nu+dim)/2.0 * jnp.log(1 + 1/nu*(x-mu).dot(solve(Sigma, x-mu)))
    return l0 + l1


def log_predic_t(x, obs, hyper_params):
    """
    Evaluating the logarithm of probability of the posterior predictive multivariate T distribution.
    The likelihood of the observation given the parameter is Gaussian distribution.
    The prior distribution is Normal Inverse Wishart (NIW) with parameters given by hyper_params.
    ---------------------------------------------------------------------------------------------
    x: array(dim)
        Data point that we want to evalute the log probability 
    obs: array(n, dim)
        Observations that the posterior distritbuion is conditioned on
    hyper_params: () 
        The set of hyper parameters of the NIW prior
    ------------------------------------------------
    * float
        Log probability of the multivariate T distribution at x
    """
    mu0, kappa0, nu0, Sigma0 = hyper_params
    n, dim = obs.shape
    # Use the prior marginal distribution if no observation
    if n==0:
        nu_t = nu0 - dim + 1 
        mu_t = mu0
        Sigma_t = Sigma0*(kappa0+1)/(kappa0*nu_t)
        return log_p_of_multi_t(x, nu_t, mu_t, Sigma_t)
    # Update the distribution using sufficient statistics
    obs_mean = jnp.mean(obs, axis=0)
    S = (obs-obs_mean).T @ (obs-obs_mean)
    nu_n = nu0 + n
    kappa_n = kappa0 + n
    mu_n = kappa0/kappa_n*mu0 + n/kappa_n*obs_mean
    Lambda_n = Sigma0 + S + kappa0*n/kappa_n*jnp.outer(obs_mean-mu0, obs_mean-mu0)
    nu_t = nu_n - dim + 1
    mu_t = mu_n
    Sigma_t = Lambda_n*(kappa_n+1)/(kappa_n*nu_t)
    return log_p_of_multi_t(x, nu_t, mu_t, Sigma_t)