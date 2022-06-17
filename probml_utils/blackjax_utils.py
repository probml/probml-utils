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
        samples = {"samples":jnp.swapaxes(states.position,0,1)}
        divergence = jnp.swapaxes(info.is_divergent, 0, 1)
     
    else: # if states.position is dict 
        samples = {}        
        for param in states.position.keys():
            ndims = len(states.position[param].shape)
            if ndims == 2:
                samples[param] = jnp.swapaxes(states.position[param], 0, 1)[:, burn_in:]  # swap n_samples and n_chains
                divergence = jnp.swapaxes(info.is_divergent[burn_in:], 0, 1)

            if ndims == 1:
                divergence = info.is_divergent
                samples[param] = states.position[param]
                
    trace_posterior = az.convert_to_inference_data(samples)
    trace_sample_stats = az.convert_to_inference_data({"diverging": divergence}, group="sample_stats")
    trace = az.concat(trace_posterior, trace_sample_stats)
    return trace

def inference_loop_multiple_chains(rng_key, kernel, initial_states, num_samples, num_chains):
    '''
    returns dict: {"states": states, "info": info}
    Visit this page for more info: https://blackjax-devs.github.io/blackjax/examples/Introduction.html
    '''
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, info = jax.vmap(kernel)(keys, states) 
        return states, {"states": states, "info": info}

    keys = jax.random.split(rng_key, num_samples)
    _, states_and_info = jax.lax.scan(one_step, initial_states, keys)

    return states_and_info
