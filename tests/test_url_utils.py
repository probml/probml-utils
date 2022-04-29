from probml_utils.url_utils import check_dead_urls

def test_dead_urls():
    links = {'1.3': 'https://github.com/probml/pyprobml/blob/master/notebooks/book1/01/iris_plot.ipynb',
    '1.4': 'https://github.com/probml/pyprobml/blob/master/notebooks/book1/01/iris_dtree.ipynb',
    '1.5': 'https://github.com/probml/pyprobml/blob/master/notebooks/book1/01/linreg_residuals_plot_broken.ipynb',
    '1.6': 'https://github.com/probml/pyprobml/blob/master/notebooks/book1/01/linreg_2d_surface_demo_broken.ipynb'}

    status = check_dead_urls(links)
    status_true = {'1.3': 0, '1.4': 0, '1.5': 1, '1.6': 1}
    assert (status == status_true)