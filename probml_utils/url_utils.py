import jax
import requests
from typing import Any
from TexSoup import TexSoup
import regex as re
import os
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app

def is_dead_url(link):
    '''
    check if given link is dead or not
    '''
    resp = requests.get(link)
    if resp.status_code != 200:
        return 1
    return 0

def check_dead_urls(urls: Any, print_dead_url=True):
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

def github_url_to_colab_url(url):
    '''
    convert github .ipynb url to colab .ipynb url
    '''
    if not (url.startswith("https://github.com")):
        raise ValueError("INVALID URL: not a Github url")
    
    if not(url.endswith(".ipynb")):
        raise ValueError("INVALID URL: not a .ipynb file")
    
    base_url_colab = "https://colab.research.google.com/github/"
    base_url_github = "https://github.com/"
    
    return url.replace(base_url_github, base_url_colab)

def colab_url_to_github_url(url):
    '''
    convert colab .ipynb url to github .ipynb url
    '''
    if not (url.startswith("https://colab.research.google.com/github")):
        raise ValueError("INVALID URL: not a colab github url")
    
    if not(url.endswith(".ipynb")):
        raise ValueError("INVALID URL: not a .ipynb file")
    
    base_url_colab = "https://colab.research.google.com/github/"
    base_url_github = "https://github.com/"
    return url.replace(base_url_colab,base_url_github)

def colab_to_githubraw_url(url):
    '''
    convert colab .ipynb url to github raw .ipynb url
    '''
    if not (url.startswith("https://colab.research.google.com/github")):
        raise ValueError("INVALID URL: not a colab github url")
    
    if not(url.endswith(".ipynb")):
        raise ValueError("INVALID URL: not a .ipynb file")
    
    base_url_colab = "https://colab.research.google.com/github/"
    base_url_githubraw = "https://raw.githubusercontent.com/"
    return url.replace(base_url_colab,base_url_githubraw).replace("blob/","").replace("tree/","")

def extract_scripts_name_from_caption(caption):
    """
    extract foo.py from ...{https//:<path/to/>foo.py}{foo.py}...
    Input: caption
    Output: ['foo.py']
    """
    py_pattern = r"\{\S+?\.py\}"
    ipynb_pattern = r"\{\S+?\.ipynb\}"

    matches = re.findall(py_pattern, str(caption)) + re.findall(ipynb_pattern, str(caption))
    extracted_scripts = []
    for each in matches:
        if "https" not in each:
            each = each.replace("{", "").replace("}", "").replace("\\_", "_")
            extracted_scripts.append(each)
    return extracted_scripts


def make_url_from_fig_no_and_script_name(
    fig_no, script_name, base_url="https://github.com/probml/pyprobml/blob/master/notebooks", book_no=1, convert_to_colab_url = True
):
    """
    create mapping between fig_no and actual_url path
    (fig_no=1.3,script_name=iris_plot.ipynb) converted to https://github.com/probml/pyprobml/blob/master/notebooks/book1/01/iris_plot.ipynb
    """
    chapter_no = int(fig_no.strip().split(".")[0])
    base_url_ipynb = os.path.join(base_url, f"book{book_no}/{chapter_no:02d}")
    if ".py" in script_name:
        script_name = script_name[:-3] + ".ipynb"
    if convert_to_colab_url:
        return github_url_to_colab_url(os.path.join(base_url_ipynb, script_name))
    return os.path.join(base_url_ipynb, script_name)


def make_url_from_chapter_no_and_script_name(
    chapter_no, script_name, base_url="https://github.com/probml/pyprobml/blob/master/notebooks", book_no=1, convert_to_colab_url = True
):
    """
    create mapping between chapter_no and actual_url path
    (chapter_no = 3,script_name=iris_plot.ipynb) converted to https://github.com/probml/pyprobml/blob/master/notebooks/book1/01/iris_plot.ipynb
    """
    base_url_ipynb = os.path.join(base_url, f"book{book_no}/{int(chapter_no):02d}")
    if script_name.strip().endswith(".py"):
        script_name = script_name[:-3] + ".ipynb"
    if convert_to_colab_url:
        return github_url_to_colab_url(os.path.join(base_url_ipynb, script_name))
    return os.path.join(base_url_ipynb, script_name)


def dict_to_csv(key_value_dict, csv_name):    
    df = pd.DataFrame(key_value_dict.items(), columns=["key", "url"])
    df.set_index(keys=["key"], inplace=True, drop=True)
    df.to_csv(csv_name)


def figure_url_mapping_from_lof(lof_file_path, csv_name, convert_to_colab_url = True, base_url = "https://github.com/probml/pyprobml/blob/master/notebooks", book_no=1):
    f'''
    create mappng of fig_no to url by parsing lof_file and save mapping in {csv_name}
    '''
    with open(lof_file_path) as fp:
        LoF_File_Contents = fp.read()
    soup = TexSoup(LoF_File_Contents)
    
    # create mapping of fig_no to list of script_name
    
    url_mapping = {}
    for caption in soup.find_all("numberline"):
        fig_no = str(caption.contents[0])
        extracted_scripts = extract_scripts_name_from_caption(str(caption))
        if len(extracted_scripts) == 1:
            url_mapping[fig_no] = make_url_from_fig_no_and_script_name(fig_no, extracted_scripts[0], convert_to_colab_url=convert_to_colab_url, base_url=base_url, book_no=book_no)
        elif len(extracted_scripts) > 1:
            url_mapping[fig_no] = make_url_from_fig_no_and_script_name(fig_no, "fig_"+fig_no.replace(".","_")+".ipynb", convert_to_colab_url= convert_to_colab_url,  base_url=base_url, book_no=book_no)
    
    if csv_name:
        dict_to_csv(url_mapping,csv_name)
    print(f"Mapping of {len(url_mapping)} urls is saved in {csv_name}")
    return url_mapping


def non_figure_notebook_url_mapping(notebooks_path, csv_name, convert_to_colab_url = True, base_url = "https://github.com/probml/pyprobml/blob/master/notebooks", book_no=1):
    f'''
    create mapping of notebook_name to url using notebooks in given path - {notebooks_path} and save mapping in {csv_name}
    '''
    url_mapping = {}
    for notebook_path in notebooks_path:
        parts = notebook_path.split("/")
        script_name = parts[-1]
        chapter_no = parts[-2]
        url = make_url_from_chapter_no_and_script_name(chapter_no,script_name,convert_to_colab_url=convert_to_colab_url, base_url=base_url, book_no=book_no)
        key = script_name.split(".")[0] # remove extension
        url_mapping[key] = url
    if csv_name:
        dict_to_csv(url_mapping,csv_name)
        print(f"Mapping of {len(url_mapping)} urls is saved in {csv_name}")
    return url_mapping

def create_firestore_db(key_path):
    cred = credentials.Certificate(key_path)
    try:
        default_app = initialize_app(cred) #this should called only once
    except ValueError:
        firebase_admin.delete_app(firebase_admin.get_app()) # delete current firebase app
        default_app = initialize_app(cred) 
    db = firestore.client()
    return db

def upload_urls_to_firestore(key_path,csv_path,level1_collection = "figures",
                             level2_document = None ,level3_collection = None):
    
    f'''
    extract key-value pair from {csv_path} and upload in firestore  database
    '''
    assert level2_document in ["book1", "book2"], "Incorrect level2_document value: possible values of level2_document should be ['book1', 'book2']"

    db = create_firestore_db(key_path)

    collection = db.collection(level1_collection).document(level2_document).collection(level3_collection)
    
    df = pd.read_csv(csv_path, dtype = str) # put dtype=str otherwise fig_no 3.30 will converted to 3.3
    
    assert sorted(df.columns) ==  ["key","url"], f"columns of {csv_path} should be only 'key' and 'url'" 
    
    print("Uploading...")
    for (key,url) in list(zip(df["key"], df["url"])):
        collection.document(key).set({"link": url})
    print(f"{len(df)} urls uploaded!")