from setuptools import find_packages, setup

setup(
    name="seq2seq-de-en",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "datasets",
        "matplotlib",
        "numpy",
        "requests",
        "tqdm",
    ],
)
