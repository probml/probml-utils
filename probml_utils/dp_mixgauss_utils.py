import jax.numpy as jnp
from jax import jit, vmap, random, lax
from collections import namedtuple
from functools import partial
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


@partial(jit, static_argnums=(1,))
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
        array(num_of_samples, dimension): mean value of Gaussian component of each sample
        array(num_of_samples, dimension, dimension): 
            variance (covariance matrix) of the Gaussian component of each sample
        array(num_of_samples, dimension): samples from the DP mixture model 
    """
    # Generating distribution parameters for each observation from the base measure
    # (redundant parameters in the same cluster are to be removed during sampling)
    key, subkey = random.split(key)
    cluster_parameters = dp_base_measure.sample(seed=subkey, sample_shape=(num_of_samples,))
    def sample_update(carry, data_index):
        key, cluster_means, cluster_covs = carry
        key, subkey = random.split(key)
        cluster_position = random.uniform(subkey, minval=0.0, maxval=data_index+dp_concentration)
        # new sample is assigned to a new cluster if the uniform random variable > data_index
        # otherwise its distribution parameter is set equal to that of an existing sample 
        _mean, _cov = lax.cond(cluster_position > data_index, 
                           lambda x: (cluster_means[data_index], cluster_covs[data_index]), 
                           lambda x: (cluster_means[x], cluster_covs[x]), 
                           cluster_position.astype(int))
        cluster_means = cluster_means.at[data_index].set(_mean)
        cluster_covs = cluster_covs.at[data_index].set(_cov)
        key, subkey = random.split(key)
        sample = random.multivariate_normal(subkey, _mean, _cov)
        return (key, cluster_means, cluster_covs), sample
    carry = (key, cluster_parameters['mu'], cluster_parameters['Sigma'])
    carry, samples = lax.scan(sample_update, carry, jnp.arange(num_of_samples))
    key, cluster_means, cluster_covs = carry
    return cluster_means, cluster_covs, samples
