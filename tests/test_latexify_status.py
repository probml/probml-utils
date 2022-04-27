import probml_utils as pml
import os

def test_latexify_disabled():
    if "LATEXIFY" in os.environ:
        os.environ.pop("LATEXIFY")
    assert(pml.is_latexify_enabled() == False)

def test_latexify_enabled():
    os.environ["LATEXIFY"] = ""     
    assert(pml.is_latexify_enabled() == True)