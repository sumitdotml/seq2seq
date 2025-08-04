from setuptools import find_packages, setup

setup(
    name="seq2seq-translation",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "datasets",
        "matplotlib",
        "numpy",
    ],
)
