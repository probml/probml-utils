# modified from
# https://github.com/Paperspace/ml-in-a-box/blob/main/ml_in_a_box.sh

#!/usr/bin/env bash


# Upgrade Pytorch and TF

pip3 install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

pip3 install --upgrade --user tensorflow tensorflow_probability

# Add JAX

pip3 install  "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

pip3 install distrax optax


# Add Other stuff
pip3 install seaborn tdqm
