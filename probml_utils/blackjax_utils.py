import arviz as az
import jax.numpy as jnp
import jax

def arviz_trace_from_states(states, info, burn_in=0):
    """
    args:
    ...........
    states: contains samples returned by blackjax model (i.e HMCState)
    info: conatins the meta info returned by blackjax model (i.e HMCinfo)
    
    returns:
    ...........
    trace: arviz trace object
    """
    if isinstance(states.position, jnp.DeviceArray):  #if states.position is array of samples 
        ndims = jnp.ndim(states.position)
        if ndims > 1:
            samples = {"samples":jnp.swapaxes(states.position,0,1)}
            divergence = jnp.swapaxes(info.is_divergent, 0, 1)
        else:
            samples = states.position
            divergence = info.is_divergent
        
    else: # if states.position is dict 
        samples = {}      
        for param in states.position.keys():
            ndims = len(states.position[param].shape)
            if ndims >= 2:
                samples[param] = jnp.swapaxes(states.position[param], 0, 1)[:, burn_in:]  # swap n_samples and n_chains
            elif ndims == 1:
                samples[param] = states.position[param]
            
        divergence = info.is_divergent  
        ndims_div = len(divergence.shape)
        if ndims_div >= 2:
            divergence = jnp.swapaxes(divergence, 0, 1)[:, burn_in:]
        elif ndims_div == 1:
            divergence = info.is_divergent
                
    trace_posterior = az.convert_to_inference_data(samples)
    trace_sample_stats = az.convert_to_inference_data({"diverging": divergence}, group="sample_stats")
    trace = az.concat(trace_posterior, trace_sample_stats)
    return trace

def inference_loop_multiple_chains(rng_key, kernel, initial_states, num_samples, num_chains):
    '''
    returns (states, info)
    Visit this page for more info: https://blackjax-devs.github.io/blackjax/examples/Introduction.html
    '''
    @jax.jit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, infos = jax.vmap(kernel)(keys, states) 
        return states, (states, infos)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_states, keys)

    return (states, infos)

def inference_loop(rng_key, kernel, initial_state, num_samples):
    '''
    returns (states, info)
    Visit this page for more info: https://blackjax-devs.github.io/blackjax/examples/Introduction.html
    '''
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return (states, infos)
