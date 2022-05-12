import nbformat as nbf


def get_ipynb_from_code(code):
    """
    Get the ipynb from the code.
    """
    nb = nbf.v4.new_notebook()
    nb["cells"] = [nbf.v4.new_code_cell(code)]
    return nb
