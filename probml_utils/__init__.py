from ._version import version as __version__
from .plotting import savefig, latexify, _get_fig_name, is_latexify_enabled
from .blackjax_utils import arviz_trace_from_states
from .pyprobml_utils import (
    hinton_diagram,
    plot_ellipse,
    convergence_test,
    kdeg,
    scale_3d,
    style3d,
)
