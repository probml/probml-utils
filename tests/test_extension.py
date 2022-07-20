import os
from probml_utils import _get_fig_name


def test_extension():
    os.environ["LATEXIFY"] = ""
    assert _get_fig_name("test.pdf") == "test_latexified.pdf"
    assert _get_fig_name("test.png") == "test_latexified.pdf"
    assert _get_fig_name("test.jpg") == "test_latexified.pdf"
    assert _get_fig_name("test") == "test_latexified.pdf"
    os.environ.pop("LATEXIFY")
    assert _get_fig_name("test.pdf") == "test.pdf"
    assert _get_fig_name("test.png") == "test.pdf"
    assert _get_fig_name("test.jpg") == "test.pdf"
    assert _get_fig_name("test") == "test.pdf"
