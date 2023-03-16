# modified from
# https://github.com/Paperspace/ml-in-a-box/blob/main/ml_in_a_box.sh
# Also works for lambdalabs

#!/usr/bin/env bash


# Upgrade Pytorch and TF and JAX

pip3 install torch torchvision torchaudio

#pip3 install --upgrade --user tensorflow tensorflow_probability

pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# jax Libraries

pip3 install distrax optax flax chex
pip3 install -Uq tfp-nightly[jax]
# https://www.tensorflow.org/probability/examples/TensorFlow_Probability_on_JAX

# github
git config --global user.email "murphyk@gmail.com"
git config --global user.name "Kevin Murphy"

#sudo snap install gh
#gh auth login  # paste personal access token

# avoid having to paste PAT more than once per day
# https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage
git config --global credential.helper 'cache --timeout 90000'

# Other stuff
pip3 install seaborn tdqm
pip install jax-tqdm
