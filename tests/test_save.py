import pytest

import os
import probml_utils as pml
import matplotlib.pyplot as plt


@pytest.mark.parametrize("latexify", [True, False])
def test_save(latexify):
    if latexify:
        os.environ["LATEXIFY"] = ""
        suffix = "_latexified"
    else:
        if "LATEXIFY" in os.environ:
            os.environ.pop("LATEXIFY")
        suffix = ""

    os.environ["FIG_DIR"] = "figures"
    pml.latexify(width_scale_factor=2, fig_height=1.5)
    plt.plot([1.0, 2.0], [3.0, 4.0])
    save_name = os.path.join(os.environ["FIG_DIR"], f"test{suffix}.pdf")
    if os.path.exists(save_name):
        os.remove(save_name)
    pml.savefig("test")
    assert os.path.exists(save_name)
