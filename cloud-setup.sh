# Cloud VM setup. Modified from
# https://github.com/Paperspace/ml-in-a-box/blob/main/ml_in_a_box.sh
# Also works for lambdalabs and TPU VM

#!/usr/bin/env bash


# Upgrade Pytorch and TF and JAX

pip3 install torch torchvision torchaudio

# for pytorch on TPU V4
export TPU_NUM_DEVICES=4

#pip3 install --upgrade --user tensorflow tensorflow_probability

pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# jax Libraries
pip3 install distrax optax flax chex einops jaxtyping jax-tqdm
pip3 install -Uq tfp-nightly[jax]
# https://www.tensorflow.org/probability/examples/TensorFlow_Probability_on_JAX

# other libraries
pip3 install seaborn tdqm scikit-learn 

# github
git config --global user.email "murphyk@gmail.com"
git config --global user.name "Kevin Murphy"


# avoid having to paste PAT more than once per day
# https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage
git config --global credential.helper 'cache --timeout 90000'

#sudo snap install gh
#gh auth login  # paste personal access token


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
#jupyter labextension install base16-mexico-light