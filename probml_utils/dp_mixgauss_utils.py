import jax.numpy as jnp
from jax import random
from collections import namedtuple
from multivariate_t_utils import log_predic_t


def dp_mixture_simu(N, alpha, H, key):
    """
    Generating samples from the Gaussian Dirichlet process mixture model.
    We set the base measure of the DP to be Normal Inverse Wishart (NIW)
    and the likelihood be multivariate normal distribution 
    ------------------------------------------------------
    N: int 
        Number of samples to be generated from the mixture model
    alpha: float
        Concentration parameter of the Dirichlet process 
    H: object of NormalInverseWishart
        Base measure of the Dirichlet process
    key: jax.random.PRNGKey
        Seed of initial random cluster
    --------------------------------------------
    * array(N):
        Simulation of cluster assignment
    * array(N, dimension):
        Simulation of samples from the DP mixture model
    * array(K, dimension):
        Simulation of mean of each cluster
    * array(K, dimension, dimension):
        Simulation of covariance of each cluster
    """
    Z = jnp.full(N, 0)
    # Sample cluster assignment from the Chinese restaurant process prior 
    CR = []
    for i in range(N):
        p = jnp.array(CR + [alpha])
        key, subkey = random.split(key)
        k = random.categorical(subkey, logits=jnp.log(p))
        # Add new cluster to the mixture 
        if k == len(CR):
            CR = CR + [1]
        # Increase the size of corresponding cluster by 1 
        else:
            CR[k] += 1
        Z = Z.at[i].set(k)
    # Sample the parameters for each component of the mixture distribution, from the base measure 
    key, subkey = random.split(key)
    params = H.sample(seed=subkey, sample_shape=(len(CR),))
    Sigma = params['Sigma'] 
    Mu = params['mu']
    # Sample from the mixture distribtuion
    subkeys = random.split(key, N)
    X = [random.multivariate_normal(subkeys[i], Mu[Z[i]], Sigma[Z[i]]) for i in range(N)]
    return Z, jnp.array(X), Mu, Sigma


def dp_cluster(T, X, alpha, hyper_params, key):
    """
    Implementation of algorithm3 of R.M.Neal(2000)
    https://www.tandfonline.com/doi/abs/10.1080/10618600.2000.10474879
    The clustering analysis using Gaussian Dirichlet process (DP) mixture model
    ---------------------------------------------------------------------------
    T: int 
        Number of iterations of the MCMC sampling
    X: array(size_of_data, dimension)
        The array of observations
    alpha: float
        Concentration parameter of the DP
    hyper_params: object of NormalInverseWishart
        Base measure of the Dirichlet process
    key: jax.random.PRNGKey
        Seed of initial random cluster
    ----------------------------------
    * array(T, size_of_data):
        Simulation of cluster assignment
    """
    n, dim = X.shape
    Zs = []
    Cluster = namedtuple('Cluster', ["label", "members"])
    # Initialize by setting all observations to cluster0
    cluster0 = Cluster(label=0, members=list(range(n)))
    # CR is set of clusters
    CR = [cluster0]
    Z = jnp.full(n, 0) 
    new_label = 1 
    for t in range(T):
        # Update the cluster assignment for every observation
        for i in range(n):
            labels = [cluster.label for cluster in CR]
            j = labels.index(Z[i])
            CR[j].members.remove(i)
            if len(CR[j].members) == 0:
                del CR[j]
            lp0 = [jnp.log(len(cluster.members)) + log_predic_t(X[i,], jnp.atleast_2d(X[cluster.members[:],]), hyper_params) for cluster in CR]
            lp1 = [jnp.log(alpha) + log_predic_t(X[i,], jnp.empty((0, dim)), hyper_params)]
            logits = jnp.array(lp0 + lp1)
            key, subkey = random.split(key)
            k = random.categorical(subkey, logits=logits)
            if k==len(logits)-1:
                new_cluster = Cluster(label=new_label, members=[i])
                new_label += 1
                CR.append(new_cluster)
                Z = Z.at[i].set(new_cluster.label)
            else:
                CR[k].members.append(i)
                Z = Z.at[i].set(CR[k].label)
        Zs.append(Z)
    return jnp.array(Zs)



