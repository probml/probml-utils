from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().strip().split("\n")

setup(
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)
