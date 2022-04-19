import os
import probml_utils as pml
import matplotlib.pyplot as plt


def test_save_latexified():
    os.environ["FIG_DIR"] = "figures"
    os.environ["LATEXIFY"] = ""
    pml.latexify(width_scale_factor=2, fig_height=1.5)
    plt.plot([1.0, 2.0], [3.0, 4.0])
    pml.savefig("test")
    assert os.path.exists("figures/test.pdf")
    
def test_save_normal():
    if "LATEXIFY" in os.environ:
        os.environ.pop("LATEXIFY")
    os.environ["FIG_DIR"] = "figures"
    plt.plot([1.0, 2.0], [3.0, 4.0])
    pml.savefig("test")
    assert os.path.exists("figures/test.png")