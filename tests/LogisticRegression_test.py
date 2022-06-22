from probml_utils.LogisticRegression import binary_loss_function, multi_loss_function, fit
import jax.numpy as jnp
import jax
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import pytest

def test_binary_loss_function():
    x = jnp.array([[1, 1, 2]])
    weights = jnp.array([-1, 0.4, -0.2])
    y = jnp.array([0.26])
    loss = 0.5832
    assert jnp.allclose(loss, binary_loss_function(weights, auxs=[x, y, 0.1])[0], rtol=1e-2)

def test_multi_loss_function():
    x = jnp.array([[1, 1, 2]])
    weights = jnp.array([
        [0, 1, 2],[0.3, 0.4, 0.5], [0.5, 0.2, -0.3]
    ])
    y = jnp.array([[0.22, 0.37, 0.41]])
    loss = 1.110
    assert jnp.allclose(loss, multi_loss_function(weights, auxs=[x, y, 0.1])[0], rtol=1e-2)

def test_fit_function():
    iris = datasets.load_iris()
    train_x = iris["data"][:, (2,3)]
    train_y = (iris["target"] == 2).astype(jnp.int32)

    #lr from scratch using jax
    weights, intercept_, coef_ = fit(train_x, train_y, 10000, lambd=0.01)

    #lr from sklearn
    lr = LogisticRegression(C=100)
    lr.fit(train_x, train_y)

    assert jnp.allclose(intercept_, lr.intercept_, rtol=1e-2)
    assert jnp.allclose(coef_.T, lr.coef_, rtol=1e-2)


