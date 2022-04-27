from logging import warning
import os
import matplotlib.pyplot as plt
import warnings

DEFAULT_WIDTH = 6.0
DEFAULT_HEIGHT = 1.5
SIZE_SMALL = 9  # Caption size in the pml book


def latexify(
    width_scale_factor=1,
    height_scale_factor=1,
    fig_width=None,
    fig_height=None,
    font_size=SIZE_SMALL,
):
    f"""
    width_scale_factor: float, DEFAULT_WIDTH will be divided by this number, DEFAULT_WIDTH is page width: {DEFAULT_WIDTH} inches.
    height_scale_factor: float, DEFAULT_HEIGHT will be divided by this number, DEFAULT_HEIGHT is {DEFAULT_HEIGHT} inches.
    fig_width: float, width of the figure in inches (if this is specified, width_scale_factor is ignored)
    fig_height: float, height of the figure in inches (if this is specified, height_scale_factor is ignored)
    font_size: float, font size
    """
    if "LATEXIFY" not in os.environ:
        warnings.warn("LATEXIFY environment variable not set, not latexifying")
        return
    if fig_width is None:
        fig_width = DEFAULT_WIDTH / width_scale_factor
    if fig_height is None:
        fig_height = DEFAULT_HEIGHT / height_scale_factor

    # use TrueType fonts so they are embedded
    # https://stackoverflow.com/questions/9054884/how-to-embed-fonts-in-pdfs-produced-by-matplotlib
    # https://jdhao.github.io/2018/01/18/mpl-plotting-notes-201801/
    plt.rcParams["pdf.fonttype"] = 42

    # Font sizes
    # SIZE_MEDIUM = 14
    # SIZE_LARGE = 24
    # https://stackoverflow.com/a/39566040
    plt.rc("font", size=font_size)  # controls default text sizes
    plt.rc("axes", titlesize=font_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=font_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=font_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=font_size)  # legend fontsize
    plt.rc("figure", titlesize=font_size)  # fontsize of the figure title

    # latexify: https://nipunbatra.github.io/blog/visualisation/2014/06/02/latexify.html
    plt.rcParams["backend"] = "ps"
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("figure", figsize=(fig_width, fig_height))


def _get_fig_name(fname_full):
    LATEXIFY = "LATEXIFY" in os.environ
    extention = ".pdf" if LATEXIFY else ".png"
    if fname_full[-4:] in [".png", ".pdf", ".jpg"]:
        fname = fname_full[:-4]
        warnings.warn(
            f"renaming {fname_full} to {fname}{extention} because LATEXIFY is {LATEXIFY}",
        )
    else:
        fname = fname_full
    return fname + extention

def is_latexify_enabled():
    '''
    returns true if LATEXIFY environment variable is set
    '''
    return "LATEXIFY" in os.environ

def savefig(f_name, tight_layout=True, tight_bbox=False, *args, **kwargs):
    if len(f_name) == 0:
        return
    if "FIG_DIR" not in os.environ:
        warnings.warn("set FIG_DIR environment variable to save figures")
        return

    fig_dir = os.environ["FIG_DIR"]
    # Auto create the directory if it doesn't exist
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    fname_full = os.path.join(fig_dir, f_name)

    print("saving image to {}".format(fname_full))
    if tight_layout:
        plt.tight_layout(pad=0)
    print("Figure size:", plt.gcf().get_size_inches())

    fname_full = _get_fig_name(fname_full)
    if tight_bbox:
        # This changes the size of the figure
        plt.savefig(fname_full, pad_inches=0.0, bbox_inches="tight", *args, **kwargs)
    else:
        plt.savefig(fname_full, pad_inches=0.0, *args, **kwargs)
