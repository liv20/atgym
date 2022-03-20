# coding=utf-8
"""
Setup for atgym.
"""

import setuptools

with open("README.md") as fh:
    long_description = fh.read()
setuptools.setup(
    name="atgym",
    version="0.0.1",
    author="Victor Li",
    author_email="vhli2020@gmail.com",
    # url="https://github.com/",
    description="RL Utilities for algorithmic trading on Yahoo Finance market data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={"": ["LICENSE.txt"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "gym>=0.15.4",
        "numpy",
        "yfinance",
        "beautifulsoup4",
        "gym",
        "matplotlib",
        "mplfinance",
        "numpy",
        "pandas",
        "requests",
        "scikit_learn",
        "setuptools",
        "stable_baselines3",
        "yfinance"
    ],
)