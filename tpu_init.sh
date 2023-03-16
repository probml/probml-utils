# Install mamba
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh

# Restart bash instance 
#exec bash

# Install Jax, tfp, and github
#pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
#pip install "jax[tpu]==0.3.17" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# v 0.3.19 crashes on tpu
#https://github.com/google/jax/issues/12550


pip install -Uq "tfp-nightly[jax]" 
#mamba install -y gh

# Jupyterlab things
mamba install -y jupyterlab
mamba install -y nodejs
mamba install -y seaborn
mamba install -y jupyter
pip install --upgrade jupyterlab-git
sudo snap install tree


# Setup git
git config --global user.email "murphyk@gmail.com"
git config --global user.name "Kevin Murphy"

# Probml
pip install optax
pip install distrax
pip install flax
pip install einops

mamba install dask

# echo alias gitgr="git log --graph --full-history --all --oneline" >> ~/.bashrc
#jupyter labextension install base16-mexico-light


#import os
#import jax

#def setup_tpu(machine):
#    # Allow splitting of multi chip TPU VM 3 into 4 machines, 2 cores per machine
#    # https://gist.github.com/skye/f82ba45d2445bb19d53545538754f9a3
#    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
#    os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
#    os.environ["TPU_VISIBLE_DEVICES"] = str(machine) # "0", .., "3"
#    print(jax.devices())

#setup_tpu(0)

