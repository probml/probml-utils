import jax
import jax.numpy as jnp
from jaxopt import LBFGS


@jax.jit
def binary_loss_function(weights, auxs):
    """
    Arguments:
        weights : parameters, shape=(no. of features, )
        auxs: list contains X, y, lambd
        X : datasets, shape (no. of examples, no. of features)
        y : targets, shape (no. of examples, )
        lambd = regularization rate

    return:
        loss - binary cross entropy loss
    """
    X, y, lambd = auxs
    m = X.shape[0]  # no. of examples
    z = jnp.dot(X, weights)
    hypothesis_x = jax.nn.sigmoid(z)
    cost0 = jnp.dot(y.T, jnp.log(hypothesis_x + 1e-7))
    cost1 = jnp.dot((1 - y).T, jnp.log(1 - hypothesis_x + 1e-7))
    regularization_cost = (lambd * jnp.sum(weights[1:] ** 2)) / (2 * m)

    return -((cost0 + cost1) / m) + regularization_cost, auxs


@jax.jit
def multi_loss_function(weights, auxs):
    """
    Arguments:
        weights : parameters, shape=(no. of features, no. of classes)
        auxs: list contains X, y, lambd
        X : datasets, shape (no. of examples, no. of features)
        y = targets, shape (no. of examples, )
        lambd = regularization rate
    return:
        loss - CrossEntropy loss
    """
    X, y, lambd = auxs
    m = X.shape[0]  # no. of examples
    z = jnp.dot(X, weights)
    hypothesis_x = jax.nn.softmax(z, axis=1)
    regularization_cost = (lambd * jnp.sum(jnp.sum(weights[1:, :] ** 2, axis=0))) / (
        2 * m
    )

    return (-jnp.sum(y * jnp.log(hypothesis_x + 1e-7)) / m + regularization_cost), auxs


def fit(X, y, max_iter=None, learning_rate=0.1, lambd=1, random_key=1, tol=1e-8):
    """
    Arguments:
        X : training dataset, shape = (no. of examples, no. of features)
        y : targets, shape = (no. of example, )
        max_iter : maximum no. of iteration algorithms run
        learning_rate : stepsize to update the parameters
        lambda : regularization rate
        random_key : unique key to generate pseudo random numbers
        tol : gradient tolerance factor for LBFGS
    returns:
        weights : parameters, shape = (no. of features, no. of classes)
        bias : intercept, shape = (no. of classes, )
        weights : coefficient, shape = (no. of features, no. of classes)

    """
    classes = jnp.unique(y)
    n_classes = len(classes)
    key = jax.random.PRNGKey(random_key)

    # adding one more feature of ones for the bias term
    X = jnp.concatenate([jnp.ones([X.shape[0], 1]), X], axis=1)

    n_f = X.shape[1]  # no. of features
    m = X.shape[0]  # no. of examples

    if max_iter is None:
        max_iter = n_f * 200

    if n_classes > 2:
        weights = jax.random.normal(key=key, shape=[n_f, n_classes])
        y = jax.nn.one_hot(y, n_classes)
        opt = LBFGS(multi_loss_function, has_aux=True, maxiter=max_iter, tol=tol)
        weights = opt.run(weights, auxs=[X, y, lambd]).params
        return weights, weights[0, :], weights[1:, :]

    elif n_classes == 2:
        weights = jax.random.normal(
            key=key,
            shape=[
                n_f,
            ],
        )
        opt = LBFGS(binary_loss_function, has_aux=True, maxiter=max_iter, tol=tol)
        weights = opt.run(weights, auxs=[X, y, lambd]).params
        return weights, weights[0], weights[1:]


def predict(weights, x):
    """
    Arguments:
        weights : Trained Parameter, shape = (no. of features, no. of classes)
        x : int or array->shape(no. of examples, no. of features)
    Return:
        pred_y : predicted class
    """
    x = jnp.concatenate([jnp.ones([x.shape[0], 1]), x], axis=1)
    z = jnp.dot(x, weights)
    if len(z.shape) > 1:
        probs_y = jax.nn.softmax(z, axis=1)
        pred_y = jnp.argmax(probs_y, axis=1)
    else:
        probs_y = jax.nn.sigmoid(z)
        pred_y = (probs_y > 0.5).astype(int)
    return pred_y


def score(weights, x, y):
    """
    Arguments:
        weights : Trained Parameter, shape = (no. of features, no. of classes)
        x : int or array->shape(no. of examples, no. of features)
        y : int or array->shape(no. of examples,)
    """
    y_pred = predict(weights, x)
    return jnp.sum(y_pred == y) / len(x)
