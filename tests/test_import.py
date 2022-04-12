import pytest


def test_import():
    # import the savefig and latexify modules
    import probml_utils
    from probml_utils import latexify, savefig

    # import other files
    from probml_utils import fisher_lda_fit
    from probml_utils import gauss_utils
    from probml_utils import gmm_lib
    from probml_utils import mix_bernoulli_lib
    from probml_utils import plotting
    from probml_utils import prefit_voting_classifier
    from probml_utils import rvm_classifier
    from probml_utils import rvm_regressor
