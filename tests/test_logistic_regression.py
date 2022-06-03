from probml_utils.logistic_regression import binary_loss_function, multi_loss_function, predict_prob, fit, init_weights
import jax.numpy as jnp
import jax


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
    x = jnp.array([[2, 3, 0.3, 0.3],
                   [1, 3, 0.7, 0.1],
                   [3, 2, 0.1, 0.7],
                   [2, 2, 0.5, 0.1]])

    params = {
        "weights": jnp.array(
            [[0.4, 0.5, 0.6],
             [0.2, 0.6, -0.2],
             [0.07, 0.26, 0.13],
             [0.13, 0.52, 0.24]]
        ),
        "bias": jnp.array([
            [0.7, -0.8, 1]
        ])
    }
    y = jnp.array([1, 0, 2, 0])

    test_multi_loss_function()
    loss = multi_loss_function(params, x, jax.nn.one_hot(y, 3), 1)

    p, l = fit(x, y, max_iter=100, learning_rate=0.01, lambd=1)
    assert jnp.allclose(l[-1], loss, rtol=1e-1)



test_lr_fit()
test_predict_prob()
test_init_weights()
test_multi_loss_function()
test_binary_loss_function()