'''
Fits Bernoulli mixture model for mnist digits using em algorithm
Author: Meduri Venkata Shivaditya, Aleyna Kara(@karalleyna)
'''

from jax.random import PRNGKey, randint
import tensorflow as tf
from probml_utils.mix_bernoulli_lib import BMM

def mnist_data(n_obs, rng_key=None):
    '''
    Downloads data from tensorflow datasets
    Parameters
    ----------
    n_obs : int
        Number of digits randomly chosen from mnist
    rng_key : array
        Random key of shape (2,) and dtype uint32
    Returns
    -------
    * array((n_obs, 784))
        Dataset
    '''
    rng_key = PRNGKey(0) if rng_key is None else rng_key

    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x = (x_train > 0).astype('int')  # Converting to binary
    dataset_size = x.shape[0]

    perm = randint(rng_key, minval=0, maxval=dataset_size, shape=((n_obs,)))
    x_train = x[perm]
    x_train = x_train.reshape((n_obs, 784))

    return x_train