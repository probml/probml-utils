from probml_utils.url_utils import check_dead_urls, github_url_to_colab_url, colab_url_to_github_url
import pytest

def test_dead_urls():
    links = {'1.3': 'https://github.com/probml/pyprobml/blob/master/notebooks/book1/01/iris_plot.ipynb',
    '1.4': 'https://github.com/probml/pyprobml/blob/master/notebooks/book1/01/iris_dtree.ipynb',
    '1.5': 'https://github.com/probml/pyprobml/blob/master/notebooks/book1/01/linreg_residuals_plot_broken.ipynb',
    '1.6': 'https://github.com/probml/pyprobml/blob/master/notebooks/book1/01/linreg_2d_surface_demo_broken.ipynb'}

    status = check_dead_urls(links)
    status_true = {'1.3': 0, '1.4': 0, '1.5': 1, '1.6': 1}
    assert (status == status_true)

def test_github_to_colab():
    links = {
        "https://github.com/probml/pyprobml/blob/master/notebooks/book1/13/mlp_1d_regression_hetero_tfp.ipynb":"https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/book1/13/mlp_1d_regression_hetero_tfp.ipynb"}

    for link in links:
        assert links[link] == github_url_to_colab_url(link)
    
    invalid_links = ["https://google.com/probml/pyprobml/blob/master/notebooks/book1/13/mlp_1d_regression_hetero_tfp.ipynb",
        "https://github.com/probml/pyprobml/blob/master/notebooks/book1/13/mlp_1d_regression_hetero_tfp.py"]

    for link in invalid_links: 
        with pytest.raises(ValueError):
            github_url_to_colab_url(link)

def test_colab_to_github():
    links = {
        "https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/book1/13/mlp_1d_regression_hetero_tfp.ipynb":"https://github.com/probml/pyprobml/blob/master/notebooks/book1/13/mlp_1d_regression_hetero_tfp.ipynb"}
    
    for link in links:
        assert links[link] == colab_url_to_github_url(link)
    
    invalid_links = ["https://github.research.google.com/github/probml/pyprobml/blob/master/notebooks/book1/13/mlp_1d_regression_hetero_tfp.ipynb",
        "https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/book1/13/mlp_1d_regression_hetero_tfp.txt"]
        
    for link in invalid_links: 
        with pytest.raises(ValueError):
            colab_url_to_github_url(link)