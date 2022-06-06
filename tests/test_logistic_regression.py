from probml_utils.logistic_regression import binary_loss_function, multi_loss_function, predict_prob, fit, init_weights
import jax.numpy as jnp
import jax
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

def test_binary_loss_function():
    x = jnp.array([[1, 2]])
    params = {
        "weights" : jnp.array([[0.4], [-0.2]]),
        "bias" : jnp.array([-1])
    }
    y = jnp.array([0.26])
    loss = 0.5832
    assert jnp.allclose(loss, binary_loss_function(params, x, y, 0.1), rtol=1e-2)

def test_multi_loss_function():
    x = jnp.array([[1, 2]])
    params = {
        "weights": jnp.array([[0.3, 0.4, 0.5], [0.5, 0.2, -0.3]]),
        "bias": jnp.array([[0, 1, 2]])
    }
    y = jnp.array([[0.22, 0.37, 0.41]])
    loss = 1.110
    assert jnp.allclose(loss, multi_loss_function(params, x, y, 0.1), rtol=1e-2)

def test_predict_prob():
    x = jnp.array([[1, 2, 4]])
    params = {
        "weights": jnp.array([[0.3], [0.1], [0.5]]),
        "bias": jnp.array([0.5])
    }
    prob = 0.95257
    assert jnp.allclose(prob, predict_prob(parameters=params, x=x), rtol=1e-2)

def test_init_weights():
    n_f = 5 #no. of features
    n_c = 3 #no. of classes

    parameters = init_weights(n_f, n_c, random_key= 1)

    assert (n_f,n_c) == parameters["weights"].shape
    assert (1, n_c) == parameters["bias"].shape


def test_lr_fit():

    iris = datasets.load_iris()
    train_x = iris["data"][:, (2,3)]
    train_y = (iris["target"] == 2).astype(jnp.int32)

    #lr from scratch using jax
    parameters, losses = fit(train_x, train_y, 10000, lambd=0.01)

    #lr from sklearn
    lr_sklearn = LogisticRegression(C=100)
    lr_sklearn.fit(train_x, train_y)

    assert jnp.allclose(parameters["bias"], lr_sklearn.intercept_, rtol=1e-2)
    assert jnp.allclose(parameters["weights"].T, lr_sklearn.coef_, rtol=1e-2)


test_multi_loss_function()
test_binary_loss_function()
test_init_weights()
test_predict_prob()
test_lr_fit()