import jax.numpy as jnp
from jax import random
from multivariate_t_utils import log_predic_t


def gibbs_gmm(T, X, alpha, K, hyper_params, key):
    """
    Implementation of the cluster analysis with a K component mixture distribution,
    using Gaussian likelihood and normalized inverse Wishart (NIW) prior
    --------------------------------------------------------------------
    T: int 
        Number of iterations of the Gibbs sampling
    X: array(size_of_data, dimension)
        The array of observations
    alpha: float
        Precision of a symmetric Dirichlet distribution, which is the pior of the mixture weights
    hyper_params: object of NormalInverseWishart
        Base measure of the Dirichlet process
    K: int
        Number of component of the mixture distribution
    key: jax.random.PRNGKey
        Seed of initial random cluster
    ----------------------------------
    * array(T, size_of_data):
        Simulation of cluster assignment
    """
    Zs = []
    n, dim = X.shape
    CR = [[] for k in range(K)]
    Z = jnp.full(n, 0)
    CR[0] = list(range(n))
    logits = jnp.ones(K)
    for t in range(T):
        print(t)
        for i in range(n):
            k_i = Z[i]
            CR[k_i].remove(i)
            for k in range(K):
                l_k = len(CR[k])
                X_k = jnp.atleast_2d(X[CR[k][:],]) if l_k>0 else jnp.empty((0, dim))
                logits = logits.at[k].set(jnp.log(l_k + alpha/K) + log_predic_t(X[i,], X_k, hyper_params))
            key, subkey = random.split(key)
            j = random.categorical(subkey, logits=logits)
            Z = Z.at[i].set(j)
            CR[j].append(i)
        Zs.append(Z)
    return jnp.array(Zs)