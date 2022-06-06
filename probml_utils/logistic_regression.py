import jax
import jax.numpy as jnp 
import optax

@jax.jit
def binary_loss_function(parameters, x, y, lambd):
    '''
    Arguments:
        parameters: a dictionary contains weights and bias
        x = datasets, shape (no. of examples, no. of features)
        y = datasets, shape (no. of examples, )
        lambd = regularization rate
    
    return:
        loss - binary cross entropy loss
    '''
    z = jnp.dot(x,parameters["weights"]) + parameters["bias"]
    hypothesis_x = jax.nn.sigmoid(z)
    cost0 = jnp.dot(y.T, jnp.log(hypothesis_x + 1e-7))
    cost1 = jnp.dot((1 - y).T, jnp.log(1 - hypothesis_x + 1e-7))
    regularizer = (lambd * jnp.sum(jnp.sum(parameters["weights"]**2)))/(2*x.shape[0])
    
    return -((cost0 + cost1)/y.shape[0])[0] + regularizer

@jax.jit
def multi_loss_function(parameters, x, y, lambd):
    """
    args:
        parameters: a dictionary contains weights and bias
        x = datasets, shape (no. of examples, no. of features)
        y = datasets, shape (no. of examples, no. of classes)
        lambd = regularization rate
    
    return:
        loss = cross-entropy loss
    """
    z = jnp.dot(x, parameters["weights"]) + parameters["bias"]
    hypothesis_x = jax.nn.softmax(z,axis=1)
    regularizer = (lambd * jnp.sum(jnp.sum(parameters["weights"]**2,axis=0)))/(2*x.shape[0])
    
    return (-jnp.sum(y*jnp.log(hypothesis_x + 1e-7))/x.shape[0] + regularizer)

def init_weights(n_f, n_c, random_key):
    """
    Arguments:
        n_f : no. of features
        n_c : no. of classes
        random_key = unique key to generate pseudo random numbers
    return:
        parameters: a dictionary contains weights and bias    
    """
    parameters = {}
    parameters["weights"] = jax.random.normal(key = jax.random.PRNGKey(random_key), shape = [n_f,n_c])
    parameters["bias"] = jnp.zeros((1,n_c))
    return parameters
    
def fit(x, y, max_iter = 1000, learning_rate = 0.1, lambd = 1, random_key = 1):
    """
    Used optax.adam optimizer for the gradient descent.
    Used jax.lax.scan to remove unnecessary loop.
    Used Auto-grad function of JAX for computing gradient of loss function.
    
    Arguments:
        x = training dataset, shape = (no. of examples, no. of features)
        y = targets, shape = (no. of example, )
        max_iter = maximum no. of iteration algorithms run
        learning_rate = stepsize to update the parameters
        lambda = regularization rate
        random_key = unique key to generate pseudo random numbers
    
    returns:
        parameters =  dictionary contains weights and bias that best fits the training dataset
    """
    n_classes = len(jnp.unique(y))
    if n_classes > 2 :
        y = jax.nn.one_hot(y,n_classes)
        parameters = init_weights(n_f = x.shape[1], n_c = y.shape[1], random_key=random_key)
        loss_and_grad_fn = jax.value_and_grad(multi_loss_function)
    elif n_classes == 2:
        parameters= {}
        parameters["bias"] = jnp.zeros(1)
        parameters["weights"] = jax.random.normal(key = jax.random.PRNGKey(random_key), shape = [x.shape[1],1])
        
        loss_and_grad_fn = jax.value_and_grad(binary_loss_function)
    
    optimizer = optax.adam(learning_rate = learning_rate)
    opt_state = optimizer.init(parameters) 

    
    def one_step(carry,loss):
        # iterative function to carry out loss, updated parameters     
        parameters = carry["parameters"]
        x,y = carry["x"],carry["y"]
        opt_state = carry["opt_state"]
        lambd = carry["lambd"]
        
        loss, grads = loss_and_grad_fn(parameters,x,y,lambd)
        
        updates, opt_state = optimizer.update(grads, opt_state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        
        carry["parameters"] = parameters
        carry["x"],carry["y"] = x,y
        carry["opt_state"] = opt_state
        
        return carry, loss
        
    losses = None
    carry = {
        "parameters"  : parameters,
        "x" : x,
        "y" : y,
        "opt_state":opt_state,
        "lambd":lambd
    }
    #  eliminate for-loops that have carry-over using lax.scan.
    last_carry, losses = jax.lax.scan(one_step, carry, xs = losses, length = max_iter)
    return last_carry["parameters"],losses

def predict_prob(parameters, x):
    '''
    Args:
    predict the probabilities of x given parameters
    '''
    z = jnp.dot(x, parameters["weights"]) + parameters["bias"]
    if parameters["weights"].shape[1] > 1:
        hypothesis_x = jax.nn.softmax(z,axis=1)
    else:

        hypothesis_x = jax.nn.sigmoid(z)
    return hypothesis_x

def score(x, y, parameters):
    # Evaluate accuracy
    n_classes = len(jnp.unique(y))
    if n_classes > 2 :
        y_pred = predict_prob(parameters,x)
        y_pred = jnp.argmax(y_pred,axis=1)
        return jnp.sum(y_pred == y)/y.shape[0]
    else:
        y_pred = predict_prob(parameters,x)
        y_pred = (y_pred > 0.5).astype(int)
        return jnp.sum(y_pred[:,0] == y)/y.shape[0]
    