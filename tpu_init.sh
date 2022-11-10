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