import jax
import requests
from typing import Any

def is_dead_url(link):
    '''
    check if given link is dead or not
    '''
    resp = requests.get(link)
    if resp.status_code != 200:
        return 1
    return 0

def check_dead_urls(urls: Any, print_dead_url=False):
    '''
    returns if urls are dead or not
    '''
    cnt=0
    mapping_values, mapping_treedef = jax.tree_flatten(urls) #pick only values (leaf noded)
    status = []
    for url in mapping_values:
        if is_dead_url(url):
            if print_dead_url:
                print(url)
            status.append(1)
            cnt+=1
        else:
            status.append(0)
    if print_dead_url:
        print(f"{cnt} dead urls detected!")
    return mapping_treedef.unflatten(status) # convert to original structure