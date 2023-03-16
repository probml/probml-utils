

pip3 install distrax optax flax chex einops
pip3 install -Uq tfp-nightly[jax]
# https://www.tensorflow.org/probability/examples/TensorFlow_Probability_on_JAX

pip3 install seaborn tdqm scikit-learn jax-tqdm

# Setup git
git config --global user.email "murphyk@gmail.com"
git config --global user.name "Kevin Murphy"

# for pytorch on TPU V4
export TPU_NUM_DEVICES=4

# Install mamba
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
# Restart bash instance 
#exec bash

mamba install -y jupyterlab
mamba install -y nodejs
mamba install -y seaborn
mamba install -y jupyter
pip install --upgrade jupyterlab-git
sudo snap install tree
mamba install dask

# echo alias gitgr="git log --graph --full-history --all --oneline" >> ~/.bashrc
#jupyter labextension install base16-mexico-light



