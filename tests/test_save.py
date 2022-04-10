import probml_utils as pml
import matplotlib.pyplot as plt


def test_save():
    pml.latexify(width_scale_factor=2, fig_height=1.5)
    plt.plot()
    pml.savefig("test.pdf")
