import jax
import requests
import multiprocessing as mp
from typing import Any

def is_dead_url(link):
    '''
    check if given link is dead or not
    '''
    resp = requests.get(link)
    if resp.status_code != 200:
        return 1
    return 0

def check_dead_urls(urls: Any):
    '''
    returns if urls are dead or not
    this method is using multiprocessing
    '''
    pool = mp.Pool(30)
    mapping_values, mapping_treedef = jax.tree_flatten(urls) #pick only values (leaf noded)
    status = pool.map(is_dead_url,mapping_values)
    return mapping_treedef.unflatten(status) # convert to original structure