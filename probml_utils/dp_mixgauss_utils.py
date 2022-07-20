import jax.numpy as jnp
from jax.scipy.special import digamma
from jax.nn import softmax
from jax import jit, vmap, random, lax
from jax.scipy.special import gammaln
from jax.numpy.linalg import slogdet, solve

# The implementation of Normal Inverse Wishart distribution is directly copied from 
# the note book of Scott Linderman: 
# 'Implementing a Normal Inverse Wishart Distribution in Tensorflow Probability'
# https://github.com/lindermanlab/hackathons/blob/master/notebooks/TFP_Normal_Inverse_Wishart.ipynb
# and 
# https://github.com/lindermanlab/hackathons/blob/master/notebooks/TFP_Normal_Inverse_Wishart_(Part_2).ipynb

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class NormalInverseWishart(tfd.JointDistributionNamed):
    def __init__(self, loc, mean_precision, df, scale, **kwargs):
        """
        A normal inverse Wishart (NIW) distribution with

        Args:
            loc:            \mu_0 in math above
            mean_precision: \kappa_0 
            df:             \nu
            scale:          \Psi 

        Returns: 
            A tfp.JointDistribution object.
        """
        # Store hyperparameters. 
        self._loc = loc
        self._mean_precision = mean_precision
        self._df = df
        self._scale = scale
        
        # Convert the inverse Wishart scale to the scale_tril of a Wishart.
        # Note: this could be done more efficiently.
        self.wishart_scale_tril = jnp.linalg.cholesky(jnp.linalg.inv(scale))

        super(NormalInverseWishart, self).__init__(dict(
            Sigma=lambda: tfd.TransformedDistribution(
                tfd.WishartTriL(df, scale_tril=self.wishart_scale_tril),
                tfb.Chain([tfb.CholeskyOuterProduct(),                 
                        tfb.CholeskyToInvCholesky(),                
                        tfb.Invert(tfb.CholeskyOuterProduct())
                        ])),
            mu=lambda Sigma: tfd.MultivariateNormalFullCovariance(
                loc, Sigma / mean_precision)
        ))

        # Replace the default JointDistributionNamed parameters with the NIW ones
        # because the JointDistributionNamed parameters contain lambda functions,
        # which are not jittable.
        self._parameters = dict(
            loc=loc,
            mean_precision=mean_precision,
            df=df,
            scale=scale
        )

    # These functions compute the pseudo-observations implied by the NIW prior
    # and convert sufficient statistics to a NIW posterior. We'll describe them
    # in more detail below.
    @property
    def natural_parameters(self):
        """Compute pseudo-observations from standard NIW parameters."""
        dim = self._loc.shape[-1]
        chi_1 = self._df + dim + 2
        chi_2 = jnp.einsum('...,...i->...i', self._mean_precision, self._loc)
        chi_3 = self._scale + self._mean_precision * \
            jnp.einsum("...i,...j->...ij", self._loc, self._loc)
        chi_4 = self._mean_precision
        return chi_1, chi_2, chi_3, chi_4

    @classmethod
    def from_natural_parameters(cls, natural_params):
        """Convert natural parameters into standard parameters and construct."""
        chi_1, chi_2, chi_3, chi_4 = natural_params
        dim = chi_2.shape[-1]
        df = chi_1 - dim - 2
        mean_precision = chi_4
        loc = jnp.einsum('..., ...i->...i', 1 / mean_precision, chi_2)
        scale = chi_3 - mean_precision * jnp.einsum('...i,...j->...ij', loc, loc)
        return cls(loc, mean_precision, df, scale)

    def _mode(self):
        r"""Solve for the mode. Recall,
        .. math::
            p(\mu, \Sigma) \propto
                \mathrm{N}(\mu | \mu_0, \Sigma / \kappa_0) \times
                \mathrm{IW}(\Sigma | \nu_0, \Psi_0)
        The optimal mean is :math:`\mu^* = \mu_0`. Substituting this in,
        .. math::
            p(\mu^*, \Sigma) \propto IW(\Sigma | \nu_0 + 1, \Psi_0)
        and the mode of this inverse Wishart distribution is at
        .. math::
            \Sigma^* = \Psi_0 / (\nu_0 + d + 2)
        """
        dim = self._loc.shape[-1]
        covariance = jnp.einsum("...,...ij->...ij", 
                               1 / (self._df + dim + 2), self._scale)
        return self._loc, covariance

class MultivariateNormalFullCovariance(tfd.MultivariateNormalFullCovariance):
    """
    This wrapper adds simple functions to get sufficient statistics and 
    construct a MultivariateNormalFullCovariance from parameters drawn
    from the normal inverse Wishart distribution.
    """
    @classmethod
    def from_parameters(cls, params, **kwargs):
        return cls(*params, **kwargs)

    @staticmethod
    def sufficient_statistics(datapoint):
        return (1.0, datapoint, jnp.outer(datapoint, datapoint), 1.0)

##############################################################################

def dp_mixgauss_sample(key, num_of_samples, dp_concentration, dp_base_measure):
    """Sampling from the Dirichlet process (DP) Gaussian mixture model.
    
    The the base measure of the DP is the Normal Inverse Wishart (NIW), 
    which is conjugate to multivariate Gaussian distribution.
    
    Args:
        key (jax.random.PRNGKey): seed of initial random cluster
        num_of_samples (int): number of samples from the mixture model
        dp_concentration(positive float): concentration parameter (alpha) of DP
        dp_base_measure (object of the class NormalInverseWishart): 
            base measure of the Dirichlet process
                        
    Returns:
        dict: means and covariance matrices for the Gaussian distribution of each cluster 
        array(num_of_samples): 
            cluter index of each datum
        array(num_of_samples, dimension): samples from the DP mixture model 
    """
    def sample_cluster_index(carry, _):
        key, cluster_sizes, num_of_clusters = carry
        key, subkey = random.split(key)
        logits = jnp.log(cluster_sizes.at[num_of_clusters].set(dp_concentration))
        new_cluster_index = random.categorical(subkey, logits)
        cluster_sizes = cluster_sizes.at[new_cluster_index].add(1)
        num_of_clusters = lax.cond(new_cluster_index==num_of_clusters,
                                   lambda x: x+1,
                                   lambda x: x,
                                   num_of_clusters)
        return (key, cluster_sizes, num_of_clusters), new_cluster_index
    carry = (key, jnp.full(num_of_samples, 0), 0)
    carry, cluster_indices = lax.scan(sample_cluster_index, carry, None, length=num_of_samples)
    key, _, num_of_clusters = carry
    # Generating distribution parameters for each observation from the base measure
    key, subkey = random.split(key)
    cluster_parameters = dp_base_measure.sample(seed=subkey, sample_shape=(num_of_clusters,))
    # Sampling
    subkeys = random.split(key, num_of_samples)
    cluster_means = cluster_parameters['mu'][cluster_indices]
    cluster_covs = cluster_parameters['Sigma'][cluster_indices]
    samples = vmap(random.multivariate_normal)(subkeys, cluster_means, cluster_covs)
    return cluster_parameters, cluster_indices, samples


@jit
def log_pdf_multi_student(datum, nu, mu, sigma):
    """Log probability density function (pdf) of the multivariate T distribution
    
    https://en.wikipedia.org/wiki/Multivariate_t-distribution
    
    Args:
        data (array(dim)): data point that we want to evaluate log probability density at
        nu (int): degree of freedom of the multivariate T distribution
        mu (array(dim)): location parameter of the multivariate T distribution
        sigma (array(dim,dim)): positive-definite real scale matrix of the multivariate T distribution

    Returns:
        float: log probability density of the multivariate T distribution at x
    """
    dim = mu.shape[0]
    # logarithm of the normalizing constant 
    log_norm = gammaln((nu+dim)/2.0) - (gammaln(nu/2.0) 
                                        + dim/2.0*(jnp.log(nu)+jnp.log(jnp.pi))
                                        + slogdet(sigma)[1])
    # logarithm of the unnormalized pdf
    log_like = -(nu+dim)/2.0 * jnp.log(1+1/nu*(datum-mu).dot(solve(sigma, datum-mu)))
    return log_norm + log_like


@jit
def log_pdf_posterior_predictive_mvn(datum, sufficient_stat, num_of_observ, prior_params):
    """Log pdf of the posterior predictive multivariate T distribution

    The distribution of data given the parameter is Gaussian,
    and the prior distribution of parameters is the normal inverse Wishart (NIW).
    
    Args:
        sufficient_stat (dict): sufficient statistics of the observations to be conditioned on
        prior_params (dict): parameters of NIW prior

    Returns:
        float: log pdf of the posterior predictive distribution
    """
    # Parameters of the prior normal inverse Wishart distribution
    mean_prior = prior_params['loc']
    kappa_prior = prior_params['mean_precision']
    df_prior = prior_params['df']
    cov_prior = prior_params['scale'] 
    # Sufficient statistics of the Gaussian likelihood 
    mean_data = sufficient_stat['mean']
    cov_data = sufficient_stat['cov']
    # Computing the posterior parameters
    nu_pos = df_prior + num_of_observ
    kappa_pos = kappa_prior + num_of_observ 
    mu_pos = kappa_prior/kappa_pos*mean_prior + num_of_observ/kappa_pos*mean_data 
    sub = mean_data-mean_prior
    lambda_pos = cov_prior + cov_data + kappa_prior*num_of_observ/kappa_pos*jnp.outer(sub, sub) 
    dim = len(mean_data)
    # Computing parameters of the posterior predictive distribution
    nu_predict = nu_pos - dim + 1 
    mu_predict = mu_pos 
    sigma_predict = lambda_pos*(kappa_pos+1)/(kappa_pos*nu_predict) 
    return log_pdf_multi_student(datum, nu_predict, mu_predict, sigma_predict) 


def gmm_gibbs(key, num_of_samples, data, precision, num_of_clusters, prior_params):
    """Gibbs sampling of the cluster assignment using Gaussian finite mixture model
    
    The prior of the parameters of Gaussian likelihood is normal inverse Wishart (NIW)

    Args:
        key (jax.random.PRNGKey): seed of initial random cluster
        num_of_samples (int): number of samples (gibbs iterations)
        data (num_of_data, dimension): array of observations
        precision (float): precision of a symmetric Dirichlet distribution,
            which is the prior of the mixture weights
        num_of_clusters (int): number of component in the mixture distribution
        prior_params (dict): parameters of the NIW prior
    
    Returns:
        array(num_of_samples, num_of_data): samples of cluster assignment
    """
    num_of_data = data.shape[0]
    @jit
    def cluster_sufficient_stats(cluster_members):
        # Computing the sufficient statistics of one cluster,
        # 'cluster_members' is an array of boolean variable indicating 
        # whether a datum is a member of the cluster
        _mean = jnp.mean(data*jnp.atleast_2d(cluster_members).T, axis=0)
        _sub = data*jnp.atleast_2d(cluster_members).T - _mean
        _cov = _sub.T @ _sub
        return {'mean':_mean, 'cov':_cov}
    # Kernel for updating the cluster assignment of each single datum
    def cluster_assign_per_datum(carry, datum_index):
        key, cluster_assign, cluster_sizes, sufficient_stats = carry
        cluster_index = cluster_assign[datum_index]
        cluster_members = cluster_assign==cluster_index
        # Removing the current datum from its current cluster 
        # before updating the sufficient statistics of this cluster
        cluster_members = cluster_members.at[datum_index].set(False)
        cluster_sizes = cluster_sizes.at[cluster_index].add(-1)
        # Updating the sufficient statistics of the current cluster of the datum
        stat = cluster_sufficient_stats(cluster_members)
        sufficient_stats['mean'] = sufficient_stats['mean'].at[cluster_index].set(stat['mean'])
        sufficient_stats['cov'] = sufficient_stats['cov'].at[cluster_index].set(stat['cov'])
        # Assigning the data point to its new cluster
        log_likes_per_cluster = vmap(log_pdf_posterior_predictive_mvn, 
                                     in_axes=(None, {'mean': 0, 'cov': 0}, 0, None))(
                                         data[datum_index], sufficient_stats, cluster_sizes, prior_params)
        logits = log_likes_per_cluster + jnp.log(cluster_sizes + precision/num_of_clusters)
        key, subkey = random.split(key)
        new_cluster_index = random.categorical(subkey, logits)
        cluster_assign = cluster_assign.at[datum_index].set(new_cluster_index)
        cluster_sizes = cluster_sizes.at[new_cluster_index].add(1)
        # Updating the sufficient statistics for the new cluster
        cluster_members = cluster_assign==new_cluster_index
        new_stat = cluster_sufficient_stats(cluster_members)
        sufficient_stats['mean'] = sufficient_stats['mean'].at[new_cluster_index].set(new_stat['mean'])
        sufficient_stats['cov'] = sufficient_stats['cov'].at[new_cluster_index].set(new_stat['cov'])
        return (key, cluster_assign, cluster_sizes, sufficient_stats), None
    # Kernel for each gibbs iteration
    def update_per_itr(carry, key):
        # Shuffling the order of the dataset 
        shuffled_indices = random.permutation(key, jnp.arange(num_of_data))
        carry, _ = lax.scan(cluster_assign_per_datum, 
                            carry, 
                            shuffled_indices)
        return carry, carry[1]
    # Initialization by assigning data using prior distribution
    key, *subkey = random.split(key, 3)
    cluster_weights = random.dirichlet(subkey[0], precision/num_of_clusters*jnp.ones(num_of_clusters))
    cluster_assign = random.categorical(subkey[1], jnp.log(cluster_weights), shape=(num_of_data,))
    cluster_sizes = vmap(lambda x: jnp.sum(cluster_assign==x))(jnp.arange(num_of_clusters))
    sufficient_stats = vmap(lambda x: cluster_sufficient_stats(cluster_assign==x))(
                            jnp.arange(num_of_clusters))
    carry = key, cluster_assign, cluster_sizes, sufficient_stats
    # Sampling
    subkeys = random.split(key, num_of_samples)
    carry, samples_of_cluster_assign = lax.scan(update_per_itr, carry, subkeys)
    return samples_of_cluster_assign


def dp_mixgauss_gibbs(key, num_of_samples, data, concentration, prior_params):
    """Gibbs sampling of the cluster assignment using Dirichlet process (DP) Gaussian mixture model
    
    This is also known as collapsed sampling of DP mixture model.
    The prior of the parameters of the Gaussian likelihood is normal inverse Wishart (NIW)

    Args:
        key (jax.random.PRNGKey): seed of initial random sampler
        num_of_samples (int): number of samples (gibbs iterations)
        data (num_of_data, dimension): array of observations
        concentration (float): concentration parameter of the DP
        prior_params (dict): parameters of the NIW prior
    
    Returns:
        array(num_of_samples, num_of_data): samples of cluster assignment
    """
    num_of_data, dim = data.shape
    @jit
    def cluster_sufficient_stats(cluster_members):
        _mean = jnp.mean(data*jnp.atleast_2d(cluster_members).T, axis=0)
        _sub = data*jnp.atleast_2d(cluster_members).T - _mean
        _cov = _sub.T @ _sub
        return {'mean':_mean, 'cov':_cov}
    # Kernel for updating the cluster assignment for each single datum 
    def cluster_assign_per_datum(carry, datum_index):
        key, cluster_assign, cluster_sizes, sufficient_stats = carry
        cluster_index = cluster_assign[datum_index]
        cluster_members = cluster_assign==cluster_index
        # Removing the current datum from its current cluster 
        # before updating the sufficient statistics of this cluster
        cluster_members = cluster_members.at[datum_index].set(False)
        cluster_sizes = cluster_sizes.at[cluster_index].add(-1)
        # Updating the sufficient statistics of the current cluster of the datum
        stat = cluster_sufficient_stats(cluster_members)
        sufficient_stats['mean'] = sufficient_stats['mean'].at[cluster_index].set(stat['mean'])
        sufficient_stats['cov'] = sufficient_stats['cov'].at[cluster_index].set(stat['cov'])
        # Updating the weights of each cluster
        log_likes_per_cluster = vmap(_log_pdf_of_nonempty_cluster, 
                                     in_axes=(None, {'mean': 0, 'cov': 0}, 0)
                                     )(data[datum_index], sufficient_stats, cluster_sizes)
        logits = log_likes_per_cluster + jnp.log(cluster_sizes)
        # Adding (temporarily )the next cluster that could be introduced, 
        # setting the cluster index to be the index of the first empty cluster 
        log_like_next_cluster = log_pdf_posterior_predictive_mvn(data[datum_index], 
                                                                {'mean': jnp.zeros(dim), 
                                                                 'cov': jnp.zeros((dim, dim))},
                                                                0,
                                                                prior_params)
        next_cluster = jnp.asarray(cluster_sizes==0).nonzero(size=1)[0][0]
        logits = logits.at[next_cluster].set(log_like_next_cluster+jnp.log(concentration))
        # Sampling the cluster index of the datum
        key, subkey = random.split(key)
        new_cluster_index = random.categorical(subkey, logits)
        cluster_assign = cluster_assign.at[datum_index].set(new_cluster_index)
        cluster_sizes = cluster_sizes.at[new_cluster_index].add(1)
        # Updating the sufficient statistics for the new cluster
        new_cluster_members = cluster_assign==new_cluster_index
        new_stat = cluster_sufficient_stats(new_cluster_members)
        sufficient_stats['mean'] = sufficient_stats['mean'].at[new_cluster_index].set(new_stat['mean'])
        sufficient_stats['cov'] = sufficient_stats['cov'].at[new_cluster_index].set(new_stat['cov'])
        return (key, cluster_assign, cluster_sizes, sufficient_stats), None
    @jit
    def _log_pdf_of_nonempty_cluster(datum, suff_stat, cluster_size):
            # Return -inf if the cluster if empty
            # otherwise run the log_pdf_posterior_predictive_mvn
            return lax.cond(cluster_size>0, 
                            lambda _: log_pdf_posterior_predictive_mvn(datum, 
                                                                       suff_stat, 
                                                                       cluster_size, 
                                                                       prior_params),
                            lambda _: -jnp.inf,
                            None)
    # Kernel for each gibbs iteration
    def update_per_itr(carry, key):
        shuffled_indices = random.permutation(key, jnp.arange(num_of_data))
        carry, _ = lax.scan(cluster_assign_per_datum, 
                                         carry, 
                                         shuffled_indices)
        return carry, carry[1]
    # Initialization using the prior Chinese restaurant process
    def chinese_restaurant_process(carry, key):
        cluster_sizes, num_of_clusters = carry
        logits = jnp.log(cluster_sizes.at[num_of_clusters].set(concentration))
        new_cluster_index = random.categorical(key, logits)
        cluster_sizes = cluster_sizes.at[new_cluster_index].add(1)
        num_of_clusters = lax.cond(new_cluster_index==num_of_clusters,
                                   lambda x: x+1,
                                   lambda x: x,
                                   num_of_clusters)
        return (cluster_sizes, num_of_clusters), new_cluster_index
    key, subkey = random.split(key)
    carry_crp = (jnp.full(num_of_data, 0), 0)
    carry_crp, cluster_assign = lax.scan(chinese_restaurant_process, 
                                         carry_crp,
                                         random.split(subkey, num_of_data))
    cluster_sizes, num_of_clusters = carry_crp
    sufficient_stats = {'mean':jnp.zeros((num_of_data, dim)), 
                        'cov':jnp.zeros((num_of_data, dim, dim))}
    stats = vmap(lambda x: cluster_sufficient_stats(cluster_assign==x))(jnp.arange(num_of_clusters))
    sufficient_stats['mean'] = sufficient_stats['mean'].at[jnp.arange(num_of_clusters)].set(stats['mean'])
    sufficient_stats['cov'] = sufficient_stats['cov'].at[jnp.arange(num_of_clusters)].set(stats['cov'])
    carry = key, cluster_assign, cluster_sizes, sufficient_stats
    subkeys = random.split(key, num_of_samples)
    # Sampling
    carry, samples_of_cluster_assign = lax.scan(update_per_itr, carry, subkeys)
    return samples_of_cluster_assign


def _expect_natural_params(natural_hyperparam1, natural_hyperparam2, fixed_covariance):
    hyper_mean = natural_hyperparam1
    hyper_covariance = natural_hyperparam2 * fixed_covariance
    expect_nat1 = jnp.linalg.solve(fixed_covariance, hyper_mean)
    expect_nat2 = jnp.trace(jnp.linalg.solve(fixed_covariance, 
                                             hyper_covariance+jnp.outer(hyper_mean, hyper_mean)))
    return expect_nat1, expect_nat2 


def dp_mixgauss_mfvi(key, num_itr, trunc_level, suff_stats, prior_params, fixed_covariance):
    """Mean field variational inference (VI) for Dirichlet process (DP) Gaussian mixture model.
    
    The target is the joint posterior distribution of the stick-breaking items of the DP, 
    and the cluter assignment variable of each observation. 
    Using the conjugate prior, the parameters of the approximate distribution 
    are updated using coordinate ascent algorithm with closed forms in each iteration.

    Args:
        num_of_itr (_type_): _description_
        trunc_level (_type_): _description_
        data (_type_): _description_
        
    Returns:

    """
    num_obs, dim_obs = suff_stats.shape
    dp_concentration, *base_natural_params = prior_params
    base_prior_1, base_prior_2 = base_natural_params
    
    def single_vi_update(category_params, _):
        # Update the parameters of the posterior beta distribution Beta(beta_param_1, beta_param_2)
        # for the brick length of each stick-breaking term
        beta_param_1 = 1 + jnp.sum(category_params, axis=0)
        _c = dp_concentration * jnp.ones(trunc_level) 
        beta_param_2 = _c.at[:-1].add(jnp.sum(jnp.flip(jnp.cumsum(jnp.flip(category_params[:,1:], axis=1), axis=1), axis=1), axis=0))
        
        # Update the distribution of the base measure for each term of of the stick-breaking 
        base_param_1 = base_prior_1 + jnp.einsum('nt,nd -> td', category_params, suff_stats)
        base_param_2 = base_prior_2 + jnp.sum(category_params, axis=0)
        
        # Compute quantities for updating the categorical distribution for the cluster assignment
        # of each datum
        digam_1 = digamma(beta_param_1)
        digam_2 = digamma(beta_param_2)
        digam_1sum2 = digamma(beta_param_1 + beta_param_2)
        expect_stick = digam_1 - digam_1sum2
        expect_one_minus_stick = digam_2 - digam_1sum2
        sum_expect_one_minus_stick = jnp.cumsum(jnp.concatenate((jnp.zeros(1), expect_one_minus_stick[:-1])))
        expect_natural_0, expect_natural_1 = vmap(_expect_natural_params, (0, 0, None))(base_param_1, 
                                                                                        base_param_2, 
                                                                                        fixed_covariance)
        
        # Update the parameters of the categorical distribution for cluster assignment of each datum
        logits_category_params = expect_stick + sum_expect_one_minus_stick \
                                 + suff_stats @ expect_natural_0.T - expect_natural_1 
        category_params = softmax(logits_category_params, axis=1)
        
        # The parameters of the mean field approximate distribution q
        params_q = {'sb_weights_params': (beta_param_1, beta_param_2),
                    'sb_base_params': (base_param_1, base_param_2),
                    'cluster_assign_params': category_params}
        
        return category_params, params_q
    
    # Initialize the categorical distribution for the cluster assignment of each data 
    # via truncated stick breaking 
    bricks = jr.beta(key, 1, dp_concentration, shape=(num_obs, trunc_level-1))
    _b0 = jnp.concatenate((bricks, jnp.ones((num_obs, 1))), axis=1)
    _b1 = jnp.concatenate((jnp.ones((num_obs, 1)), jnp.cumprod(1.-bricks, axis=1)), axis=1)
    category_params = _b0 * _b1
    
    # Train
    _, params_q = lax.scan(single_vi_update, category_params, None, num_itr)
    
    return params_q